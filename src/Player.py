import chess
import numpy as np


class Player:
    def __init__(self, is_white=True, name='Player'):
        self.name = name
        self.is_white = is_white

    def __str__(self):
        return self.name

    def next_move(self, board):
        moves = list(board.legal_moves)
        move_idx = np.random.randint(len(moves))
        return moves[move_idx]
