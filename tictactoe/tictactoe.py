class TicTacToe:
    def __init__(self):
        """
        Initializes the TicTacToe environment with a fixed board size.
        """
        self.size = 3  # 3x3 grid

    def startState(self, first="x"):
        """
        Returns the initial game state with an empty board and the specified starting player.
        """
        board = [None] * 9  # Initialize board with 9 empty cells
        return (tuple(board), first)

    def actions(self, state):
        """
        Returns a list of available (empty) cell indices where a move can be made.
        """
        board, _ = state
        return [i for i, cell in enumerate(board) if cell is None]
    
    def get_valid_move(self, state):
        """
        Prompts the user for a valid move until a correct one is entered.
        """
        while True:
            try:
                move = int(input("Your move: "))
                if move in self.actions(state):
                    return move - 1
            except ValueError:
                pass
            print("Invalid move. Try again.")

    def enact(self, state, action):
        """
        Applies an action to the board and returns the resulting game state.
        """
        board, player = state
        board = list(board)
        board[action] = "X" if player==1 else "O" # Place player's symbol on the chosen cell

        # Alternate to the next player
        return ((tuple(board), -player))

    def isEnd(self, state):
        """
        Checks whether the game has ended, either by win or draw.
        """
        board, _ = state
        # End if there's a winner or the board is full
        return self.check_winner(board) is not None or all(cell is not None for cell in board)

    def reward(self, state):
        """
        Returns the game reward: +1 for X win, -1 for O win, 0 for draw or ongoing game.
        """
        winner = self.check_winner(state[0])
        if winner == "X":
            return 1.0
        elif winner == "O":
            return -1.0
        return 0.0  # No winner

    def player(self, state):
        """
        Returns the current player to move.
        """
        return state[1]

    def finalPrint(self, result):
        """
        Prints the entire path of the game and the final reward.
        """
        reward, path = result
        print("Moves:")
        for board, player in path:
            self.print_board(board)
            print(f"Next player: {player}\n")
        print(f"\nFinal reward: {reward}")

    def print_board(self, state):
        """
        Prints the current board state in a 3x3 grid.
        """
        board, _ = state
        for i in range(0, 9, 3):
            # Represent empty cells with '.', otherwise use player symbol
            row = ["." if cell is None else cell for cell in board[i:i+3]]
            print(" ".join(row))
        print()

    def check_winner(self, board):
        """
        Checks all winning line combinations and returns the winning symbol, if any.
        """
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
            (0, 4, 8), (2, 4, 6)              # Diagonals
        ]
        for a, b, c in lines:
            # Return the winner if all cells in the line match and are not empty
            if board[a] is not None and board[a] == board[b] == board[c]:
                return board[a]
        return None  # No winner found