from players.mcts import MCTS
from players.imitator import Imitator
from alphazero.alphazero_player import AlphaZeroPlayer
# from alphazero_player import AlphaZeroPlayer
from othello.othello_game import OthelloGame
from othello.othello_visualizer import play_interactive

BOARD_SIZE = 8

# AI player factory
AI_PLAYERS = {
    'm': ('MCTS', lambda game: MCTS()),
    'i': ('Imitator', lambda game: Imitator()),
    'z': ('AlphaZero', lambda game: AlphaZeroPlayer(game) if AlphaZeroPlayer.is_available() 
          else print("AlphaZero not available, using MCTS") or MCTS())
}

def get_ai_player(game):
    """Get AI player from user input"""
    options = ', '.join([f"'{k}' = {v[0]}" for k, v in AI_PLAYERS.items()])
    print(f"Select AI: {options}")
    
    while True:
        ai_type = input("AI type: ").strip().lower()
        if ai_type in AI_PLAYERS:
            name, factory = AI_PLAYERS[ai_type]
            print(f"Loading {name}...")
            return factory(game)
        print(f"Invalid choice. Use: {', '.join(AI_PLAYERS.keys())}")

def play_human_vs_ai(game, ai, human_player=1):
    """Play human vs AI game"""
    state = game.startState(1)
    play_interactive(game, state, 
                    lambda s: s[1] == human_player, 
                    ai.choose_move)

def play_human_vs_human(game):
    """Play human vs human game"""
    state = game.startState(1)
    play_interactive(game, state, 
                    lambda s: True, 
                    lambda g, s: None)

def play_ai_vs_ai(game, ai1, ai2, num_games=100):
    """Play AI vs AI games and show results"""
    results = {"1": 0, "-1": 0, "draw": 0}
    
    print(f"Playing {num_games} games...")
    
    for i in range(1, num_games + 1):
        state = game.startState(1)
        current_ai = ai1 if i % 2 == 1 else ai2  # Alternate starting player
        
        while not game.isEnd(state):
            if state[1] == 1:
                action = ai1.choose_move(game, state)
            else:
                action = ai2.choose_move(game, state)
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
    """Main game loop"""
    game = OthelloGame(BOARD_SIZE)
    
    print("🔴 Othello Game 🔴")
    print("=" * 30)
    
    # Game mode selection
    modes = {
        'h': 'Human vs AI',
        'a': 'AI vs Human', 
        '2': 'Human vs Human',
        's': 'AI vs AI'
    }
    
    print("Game modes:")
    for key, desc in modes.items():
        print(f"  '{key}' = {desc}")
    
    while True:
        mode = input("Select mode: ").strip().lower()
        
        if mode in ['h', 'a']:
            ai = get_ai_player(game)
            human_player = 1 if mode == 'h' else -1
            play_human_vs_ai(game, ai, human_player)
            break
            
        elif mode == '2':
            play_human_vs_human(game)
            break
            
        elif mode == 's':
            print("\nAI Player 1:")
            ai1 = get_ai_player(game)
            print("\nAI Player 2:")
            ai2 = get_ai_player(game)
            
            num_games = get_num_games()
            play_ai_vs_ai(game, ai1, ai2, num_games)
            break
            
        else:
            print(f"Invalid mode. Use: {', '.join(modes.keys())}")

if __name__ == "__main__":
    main()