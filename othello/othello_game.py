from __future__ import print_function
import sys
sys.path.append('..')
# from othello_logic import Board
from .othello_logic import Board
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

class OthelloGame():#Game):
    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }

    @staticmethod
    def getSquarePiece(piece):
        return OthelloGame.square_content[piece]

    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n + 1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n*self.n:
            return (board, -player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action/self.n), action%self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves(player)
        if len(legalMoves)==0:
            valids[-1]=1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n*x+y]=1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)
        if b.has_legal_moves(player):
            return 0
        if b.has_legal_moves(-player):
            return 0
        if b.countDiff(player) > 0:
            return 1
        return -1

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.countDiff(player)

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y+1, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y+1, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                print(OthelloGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")

    # ------------------ Implemented by me ------------------

    def get_valid_move(self, state):
        """
        Prompts the user for a valid move (row,col) until a correct one is entered.
        Returns the move as a flat index.
        """
        board_size = self.n
        moves = [a for a in self.actions(state)]
        print("Legal moves:", [(m//board_size+1, m%board_size+1) for m in moves])
        while True:
            try:
                move_str = input("Your move (row,col): ")
                row_str, col_str = move_str.split(",")
                row = int(row_str.strip()) - 1
                col = int(col_str.strip()) - 1
                move = row * board_size + col
                if move in moves:
                    return move
            except (ValueError, IndexError):
                pass
            print("Invalid move. Try again.")
    
    def startState(self, first_player=1):
        board = self.getInitBoard()
        return (board, first_player)

    def actions(self, state):
        board, player = state
        valids = self.getValidMoves(board, player)
        return [i for i, valid in enumerate(valids) if valid == 1]

    def enact(self, state, action):
        board, player = state
        next_board, next_player = self.getNextState(board, player, action)
        return (next_board, next_player)

    def isEnd(self, state):
        board, player = state
        return self.getGameEnded(board, player) != 0

    def reward(self, state):
        board, player = state
        result = self.getGameEnded(board, player)
        return float(result)

    def player(self, state):
        return state[1]

    # def print_board(self, state):
    #     board, _ = state
    #     self.display(board)

    def print_board(self, state):
        """
        Visualizes the Othello board and waits for the user to click a valid move.
        Returns the move as a flat index.
        """
        board, player = state
        n = self.n
        valids = self.getValidMoves(board, player)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_xticks(np.arange(n+1))
        ax.set_yticks(np.arange(n+1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, which='both', color='black', linewidth=1)
        ax.set_facecolor((0.0, 0.6, 0.0))

        piece_colors = {1: 'white', -1: 'black'}
        alpha_color = {1: (1.0, 1.0, 1.0, 0.4), -1: (0.0, 0.0, 0.0, 0.4)}

        for y in range(n):
            for x in range(n):
                piece = board[y][x]
                cx, cy = x + 0.5, n - y - 0.5

                if piece in piece_colors:
                    ax.add_patch(patches.Circle((cx, cy), 0.4, color=piece_colors[piece], ec='black'))
                idx = y * n + x
                if valids[idx] == 1 and piece == 0:
                    ax.add_patch(patches.Circle((cx, cy), 0.4, color=alpha_color[player], linewidth=0))

        coord_text = ax.text(0.05, 1.01, '', transform=ax.transAxes, fontsize=12)

        move_selected = {'index': None}

        def on_motion(event):
            if event.inaxes != ax:
                coord_text.set_text('')
                fig.canvas.draw_idle()
                return
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < n and 0 <= y < n:
                board_y = n - y
                coord_text.set_text(f"Hovering: ({board_y},{x + 1})")  # 1-indexed row,col
            else:
                coord_text.set_text('')
            fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes != ax:
                return
            x, y = int(event.xdata), int(event.ydata)
            row, col = n - y - 1, x
            idx = row * n + col
            if 0 <= row < n and 0 <= col < n and valids[idx] == 1:
                move_selected['index'] = idx
                plt.close(fig)

        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_press_event', on_click)

        ax.set_aspect('equal')
        plt.title(f"Othello - {'White' if player == 1 else 'Black'} to move")
        plt.tight_layout()
        plt.show()

        return move_selected['index']


# game = OthelloGame(8)
# state = game.startState()
# move = game.show_board(state)
# print("You clicked move:", move)
# state = game.enact(state, move)

