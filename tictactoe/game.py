class Game:
    def startState(self, first=None):
        """
        Returns the initial game state.
        Optionally accepts the first player (e.g., "X" or "O").
        """
        raise NotImplementedError

    def actions(self, state):
        """
        Returns a list of legal actions available from the given state.
        """
        raise NotImplementedError

    def enact(self, state, action):
        """
        Applies an action to the given state and returns the resulting next state.
        Must NOT mutate the original state.
        """
        raise NotImplementedError

    def isEnd(self, state):
        """
        Returns True if the given state is terminal (game over), otherwise False.
        """
        raise NotImplementedError

    def reward(self, state):
        """
        Returns a numeric reward from the perspective of the player who just moved.
        For 2-player zero-sum: +1 (win), -1 (loss), 0 (draw).
        """
        raise NotImplementedError

    def player(self, state):
        """
        Returns the current player to move in the given state.
        For example: "X", "O", 1, 2, etc.
        """
        raise NotImplementedError

    def print_board(self, state):
        """
        (Optional) Prints a human-readable representation of the board for debugging or CLI play.
        """
        raise NotImplementedError
