import numpy as np
import time
# Assuming OthelloGame and getValidMoves are available from othello.othello_game

# --- Text-based Board Display Function ---
def print_board(board, player, valid_moves=None, message=None):
    """
    Display the Othello board state as plain text, including valid moves.
    Player 1 is 'W' (White), Player -1 is 'B' (Black).
    Valid moves are marked with the player's initial ('w' or 'b') for guidance.
    Coordinates are 1-based (row first, then column).
    """
    n = board.shape[0]  # Assuming board is an nxn numpy array
    
    # 1. Header: Current Player and Message
    player_char_upper = 'W' if player == 1 else 'B'
    player_char_lower = 'w' if player == 1 else 'b'
    player_name = 'White' if player == 1 else 'Black'

    print("\n" + "=" * 35)
    print(f"| Othello - **{player_name} ({player_char_upper})** to move |")
    print("=" * 35)
    
    if message:
        print(f"GUIDANCE: {message}")
        print("-" * 35)

    # 2. Column labels (1 to N)
    col_labels = " ".join([str(i + 1) for i in range(n)])
    print(f"  {col_labels}")
    
    # 3. Board rows
    for r in range(n):
        row_output = f"{r + 1} "  # Row label (1 to N)
        for c in range(n):
            idx = r * n + c
            piece = board[r, c]
            
            if piece == 1:
                char = 'W'  # White
            elif piece == -1:
                char = 'B'  # Black
            elif valid_moves is not None and valid_moves[idx] == 1:
                char = player_char_lower  # Valid move hint (e.g., 'w' or 'b')
            else:
                char = '.'  # Empty
            
            row_output += f"{char} "
        print(row_output)
    
    # 4. Legend
    print("-" * 35)
    print(f"Legend: W=White, B=Black, {player_char_lower}=Valid Move, .=Empty")
    print("To move, enter 'row col' (e.g., 3 4) or index 0-63.")
    print("-" * 35)

# --- Simplified Interactive Text Play Module ---
def play_interactive_text(game, initial_state, is_human_turn_fn, choose_ai_move_fn, guidance=None, ai_move_pause=0):
    """
    Run an interactive Othello game loop with text-based board display and number input.
    """
    n = game.n
    current_state = initial_state
    
    def get_human_move(board, player):
        """Prompt human player for a move and validate."""
        valids = game.getValidMoves(board, player)
        valid_indices = np.where(valids[:-1] == 1)[0] # Exclude pass
        
        while True:
            try:
                # Prompt for a move (either row,col or a single index 0-63)
                move_input = input("Enter move: ").strip()
                
                parts = move_input.split()
                
                if len(parts) == 2:
                    # Row Col input (1-based)
                    row = int(parts[0]) - 1
                    col = int(parts[1]) - 1
                    idx = row * n + col
                elif len(parts) == 1:
                    # Single index input (0-based)
                    idx = int(parts[0])
                    row, col = divmod(idx, n)
                else:
                    raise ValueError("Invalid move format.")
                
                # 1. Validate range
                if not (0 <= row < n and 0 <= col < n):
                    print(f"Coordinates out of range (1-{n}).")
                    continue
                
                # 2. Validate move legality
                if valids[idx] == 1:
                    return idx  # Valid move index
                else:
                    print(f"({row+1},{col+1}) or index {idx} is NOT a valid move. Valid indices: {valid_indices}")
            
            except ValueError:
                print("Invalid input. Please enter 'row col' (e.g., 3 4) or a single index (0-63).")
            except Exception as e:
                print(f"An error occurred: {e}. Try again.")

    def has_valid_moves(state):
        """Check if the current player has any valid moves."""
        board, player = state
        valids = game.getValidMoves(board, player)
        return np.any(valids[:-1]) # Check for non-pass moves

    # Main game loop
    while not game.isEnd(current_state):
        board, player = current_state
        player_char = 'W' if player == 1 else 'B'
        
        # Get valid moves and guidance message
        valids = game.getValidMoves(board, player)
        message = guidance(current_state) if guidance and is_human_turn_fn(current_state) else None
        
        # Display board and guidance
        print_board(board, player, valids, message)
        
        # Handle move (Pass, Human, or AI)
        if not has_valid_moves(current_state):
            print(f"Player {player_char} has no moves. **Passing** to opponent.")
            action = n * n # Pass move
        elif is_human_turn_fn(current_state):
            action = get_human_move(board, player)
        else:
            print(f"\nAI ({player_char}) is thinking...")
            t0 = time.time()
            action = choose_ai_move_fn(current_state)
            row, col = divmod(action, n)
            if action != n * n:
                print(f"AI plays **({row+1}, {col+1})** (index {action}) in {time.time() - t0:.2f}s.")
            else:
                print(f"AI has no moves and **passes** in {time.time() - t0:.2f}s.")
            time.sleep(1) # Optional: Pause for effect after AI move

        # Enact the move
        current_state = game.enact(current_state, action)

    # --- Game Over ---
    board, _ = current_state
    
    # Final board state
    print_board(board, current_state[1], message="Game Over!")

    white = np.sum(board == 1)
    black = np.sum(board == -1)
    
    # Announce result
    if white > black:
        result_msg = f"🏆 **WHITE WINS!!!** (W: {white} vs B: {black}) 🏆"
    elif black > white:
        result_msg = f"🏆 **BLACK WINS!!!** (B: {black} vs W: {white}) 🏆"
    else:
        result_msg = f"🤝 **Draw!** (W: {white} vs B: {black}) 🤝"

    print("\n" + "=" * 40)
    print(result_msg)
    print("=" * 40 + "\n")