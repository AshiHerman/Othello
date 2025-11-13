# Othello Game with AI Players

A comprehensive implementation of the classic board game **Othello** (also known as Reversi) featuring multiple AI player types and interactive gameplay. Includes all combinations of AI and human gameplay.

## Features

🎮 **Multiple AI Strategies:**
- **AlphaZero** - Deep reinforcement learning AI using neural networks and Monte Carlo Tree Search
- **MCTS (Monte Carlo Tree Search)** - Classical tree search algorithm for game playing
- **Imitator** - Neural network trained to mimic expert human player moves

🎯 **Game Modes:**
- **Human vs AI** - Play as White or Black against an AI opponent
- **AI vs AI** - Watch two AI players compete
- **Human vs Human** - Local multiplayer mode

🎨 **Visualization Options:**
- **GUI Mode** - Interactive matplotlib-based board display with click-to-play
- **Text Mode** - Console-based gameplay with coordinate input
- **Guidance Mode** - AI recommendations and win probability estimates (experimental)

## Project Structure

```
├── othello/                    # Core Othello game implementation
│   ├── othello_game.py        # Game state and rules
│   ├── othello_logic.py       # Board logic and move validation
│   ├── othello_players.py     # Player base classes
│   ├── othello_visualizer.py  # GUI visualization (matplotlib)
│   └── othello_text_visualizer.py  # Text-based visualization
│
├── players/                    # AI player implementations
│   ├── player.py              # Base Player class
│   ├── az.py                  # AlphaZero player
│   ├── mcts.py                # MCTS player
│   └── imit.py                # Imitator player
│
├── imitator/                  # Neural network imitator model
│   ├── make_layers.py         # Neural network architecture
│   ├── train.py               # Model training script
│   ├── use_model.py           # Model inference utilities
│   └── model_saves/           # Pre-trained model weights
│
├── alphazero/                 # AlphaZero framework
│   ├── Coach.py               # Self-play training
│   ├── MCTS.py                # MCTS implementation
│   ├── NeuralNet.py           # Neural network wrapper
│   └── othello/               # Othello-specific AlphaZero code
│
├── parser/                    # Data processing utilities
│   ├── parser.py              # Parse game records
│   └── make_state.py          # Convert games to training states
│
├── main.py                    # Main entry point for gameplay
└── guidance.py                # AI guidance and recommendations
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Dependencies

The project requires several packages. Install them using:

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- **numpy** - Array operations
- **torch** - PyTorch deep learning framework
- **matplotlib** - GUI visualization
- **tensorflow/keras** - (optional, for some AlphaZero features)

### Quick Setup

1. Clone or download the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main program: `python main.py`

## Usage

### Running the Game

Start the interactive game menu:

```bash
python main.py
```

### Game Mode Selection

The program will guide you through several choices:

1. **Visualization Mode:**
   - `v` = Visual mode (GUI with matplotlib)
   - `t` = Textual mode (console-based)

2. **Game Mode:**
   - `h` = Human vs AI (you play as Black, AI as White)
   - `a` = AI vs Human (you play as White, AI as Black)
   - `2` = Human vs Human (local multiplayer)
   - `s` = AI vs AI (watch two AIs play)

3. **AI Selection (for AI-involved modes):**
   - `m` = MCTS (100 simulations per move)
   - `i` = Imitator (neural network model)
   - `z` = AlphaZero (25 simulations per move)

### Example Gameplay

```
🔴 Othello Game 🔴
==============================
Choose Visualization Mode:
  'v' = Visual (Matplotlib GUI)
  't' = Textual (Plain Console)
Select visualization mode: t
------------------------------
Choose Game Mode:
  'h' = Human vs AI
  'a' = AI vs Human
  '2' = Human vs Human
  's' = AI vs AI (Silent Run)
Select mode: h
Select AI: 'm' = MCTS, 'i' = Imitator, 'z' = AlphaZero
AI type: z
Loading AlphaZero...
```

### Making Moves

**GUI Mode (Visual):**
- Click on highlighted valid move squares to place your piece

**Text Mode:**
- Enter coordinates as `row col` (1-8, e.g., `3 4`)
- Or enter linear index `0-63` for the board position
- Valid moves are marked with lowercase letters (`w` for White, `b` for Black)

## AI Players Explained

### AlphaZero
- **Method:** Deep reinforcement learning with neural networks + MCTS
- **Strength:** Strongest player, uses self-play training
- **Speed:** Configurable (25 simulations per move by default)
- **Model:** Pre-trained 8x8 Othello model included

### MCTS (Monte Carlo Tree Search)
- **Method:** Tree-based search with random playouts
- **Strength:** Strong and reliable
- **Speed:** Configurable (100 simulations by default)
- **Algorithm:** Upper Confidence bounds applied to Trees (UCT)

### Imitator
- **Method:** Convolutional neural network trained on expert games
- **Strength:** Fast, reasonable strength
- **Speed:** Very fast (single forward pass)
- **Training:** Learns patterns from human expert play records

## Configuration

Edit `main.py` to adjust game parameters:

```python
BOARD_SIZE = 8              # Board dimensions (8x8 standard)
MCTS_SIMS = 100             # Monte Carlo Tree Search simulations
AZ_SIMS = 25                # AlphaZero MCTS simulations
```

## Core Game Logic

### Board Representation
- **Size:** 8x8 (standard Othello)
- **Encoding:** Numpy array where:
  - `1` = White piece
  - `-1` = Black piece
  - `0` = Empty square

### Piece Colors
- **White (W):** Player 1
- **Black (B):** Player -1

### Rules
- Players alternate placing pieces on empty squares
- Valid moves must flip at least one opponent piece
- If no valid moves available, the player passes
- Game ends when both players pass consecutively or no empty squares remain
- Winner is determined by piece count (more pieces = win)

## Advanced Features

### Guidance Mode
The `guidance.py` module provides move recommendations and win probability estimates using both AlphaZero and Imitator models. Enable guidance in Human vs AI games to see:
- Recommended next moves from both AI models
- Estimated win probability for suggested moves
- When both models agree on the best move

### Model Training
The `imitator/` directory contains code for training new Imitator models on game records:
- `parser.py` - Parse game files and extract board states
- `train.py` - Train neural network on game data
- `use_model.py` - Visualize model predictions as heatmaps

## Troubleshooting

**ImportError for AlphaZero:**
- Ensure `alphazero/` directory structure is intact
- Verify PyTorch is properly installed: `pip install torch`

**Model loading fails:**
- Check that pre-trained model files exist in `imitator/model_saves/`
- Verify `alphazero/pretrained_models/othello/` contains model files

**GUI not displaying:**
- Ensure matplotlib is installed: `pip install matplotlib`
- Some systems may require additional display setup for matplotlib


---

**Enjoy playing Othello!** 🎮