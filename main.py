from players.mcts import MCTS
from players.imit import Imitator
from players.az import AlphaZero
from othello.othello_game import OthelloGame
from othello.othello_visualizer import play_interactive
from othello.othello_text_visualizer import play_interactive_text
from guidance import *

BOARD_SIZE = 8
MCTS_SIMS = 100
AZ_SIMS = 25

# AI player factory
AI_PLAYERS = {
    'm': ('MCTS', lambda : MCTS(MCTS_SIMS)),
    'i': ('Imitator', lambda : Imitator()),
    'z': ('AlphaZero', lambda : AlphaZero(AZ_SIMS))
}

def get_ai_player():
    """Get AI player from user input"""
    options = ', '.join([f"'{k}' = {v[0]}" for k, v in AI_PLAYERS.items()])
    print(f"Select AI: {options}")
    
    while True:
        ai_type = input("AI type: ").strip().lower()
        if ai_type in AI_PLAYERS:
            name, factory = AI_PLAYERS[ai_type]
            print(f"Loading {name}...")
            return factory()
        print(f"Invalid choice. Use: {', '.join(AI_PLAYERS.keys())}")

# Modified helper function:

# Added 'runner_fn' as a new argument
def play_human_vs_ai(game, ai, human_player=1, guidance=0, runner_fn=play_interactive): 
    """Play human vs AI game, using the specified runner function."""
    state = game.startState(1)
    # Ensure show_probs is available if guidance is 1
    func = show_probs if guidance else lambda s: None 
    
    # Now calls the selected runner_fn (either play_interactive or play_interactive_text)
    runner_fn(game, state, 
              lambda s: s[1] == human_player, 
              ai.choose_move,
              # ai_move_pause is only relevant for GUI, but harmless for text
              ai_move_pause=1 if runner_fn == play_interactive else 0,
              guidance=func)

# Added 'runner_fn' as a new argument
def play_human_vs_human(game, runner_fn=play_interactive):
    """Play human vs human game, using the specified runner function."""
    state = game.startState(1)
    
    # Now calls the selected runner_fn
    runner_fn(game, state, 
              lambda s: True, 
              lambda s: None,
              ai_move_pause=0) # ai_move_pause is ignored since there's no AI

def play_ai_vs_ai(game, ai1, ai2, num_games=100):
    """Play AI vs AI games and show results"""
    results = {"1": 0, "-1": 0, "draw": 0}
    
    print(f"Playing {num_games} games...")
    
    for i in range(1, num_games + 1):
        state = game.startState(1)
        
        while not game.isEnd(state):
            if state[1] == 1:
                action = ai1.choose_move(state)
            else:
                action = ai2.choose_move(state)
            state = game.enact(state, action)
        
        # Record result
        result = game.getGameEnded(state[0], 1)
        if result == 1:
            results["1"] += 1
        elif result == -1:
            results["-1"] += 1
        else:
            results["draw"] += 1
        
        # Progress update
        if i % max(1, num_games // 10) == 0:
            print(f"Progress: {i}/{num_games} games completed")
    
    # Final results
    print(f"\n{'='*40}")
    print(f"Results after {num_games} games:")
    print(f"  Player 1 wins: {results['1']} ({results['1']/num_games*100:.1f}%)")
    print(f"  Player -1 wins: {results['-1']} ({results['-1']/num_games*100:.1f}%)")
    print(f"  Draws: {results['draw']} ({results['draw']/num_games*100:.1f}%)")

def get_num_games():
    """Get number of games from user input"""
    while True:
        try:
            n = input("How many games? (default: 50): ").strip()
            return int(n) if n else 50
        except ValueError:
            print("Please enter a valid number")

def main():
    """Main game loop with mode selection and visualization choice."""
    game = OthelloGame(BOARD_SIZE)
    
    print("🔴 Othello Game 🔴")
    print("=" * 30)
    
    # --- New: Choose Visualization Mode ---
    print("Choose Visualization Mode:")
    print("  'v' = Visual (Matplotlib GUI)")
    print("  't' = Textual (Plain Console)")

    while True:
        vis_mode = input("Select visualization mode: ").strip().lower()
        if vis_mode == 'v':
            # Use the original GUI function
            interactive_runner = play_interactive 
            # Note: The GUI runner is designed to use mouse clicks, so AI vs AI is usually done without it.
            print("Visual mode selected. Note: AI vs AI mode will run without the GUI loop.")
            break
        elif vis_mode == 't':
            # Use the new Text function
            interactive_runner = play_interactive_text 
            print("Textual mode selected.")
            break
        else:
            print("Invalid choice. Use 'v' or 't'.")
    
    print("-" * 30)
    
    # Game mode selection (H vs AI, H vs H, AI vs AI)
    modes = {
        'h': 'Human vs AI',
        'a': 'AI vs Human', 
        '2': 'Human vs Human',
        's': 'AI vs AI (Silent Run)' # Changed for clarity, as this is typically non-interactive
    }
    for k, v in modes.items():  
        print(f"  '{k}' = {v}")
    
    while True:
        mode = input("Select mode: ").strip().lower()
        
        if mode in ['h', 'a']:
            ai = get_ai_player()
            guidance = 1 if input("Do you want guidance [y/n]: ").strip().lower() == 'y' else 0
            human_player = 1 if mode == 'h' else -1
            
            # --- Key Change: Use the selected runner function ---
            play_human_vs_ai(game, ai, human_player, guidance, interactive_runner) 
            break
            
        elif mode == '2':
            # --- Key Change: Use the selected runner function ---
            play_human_vs_human(game, interactive_runner) 
            break
            
        elif mode == 's':
            # AI vs AI is usually done without an interactive board
            ai1 = get_ai_player()
            ai2 = get_ai_player()
            num_games = get_num_games()
            play_ai_vs_ai(game, ai1, ai2, num_games)
            break
            
        else:
            print(f"Invalid mode. Use: {', '.join(modes.keys())}")

if __name__ == "__main__":
    main()