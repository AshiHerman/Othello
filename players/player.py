class Player:
    def __init__(self, name: str):
        self.name = name

    def select_move(self, board):
        """
        Select a move given the current board state.
        Should be implemented by subclasses.
        """
        raise NotImplementedError("select_move must be implemented by subclasses")

    def __str__(self):
        return f"Player({self.name})"