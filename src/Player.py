import chess
import numpy as np


class Player:
    def __init__(self,
                 is_white=True,
                 name='Player',
                 wins=0,
                 losses=0,
                 draws=0):
        self.name = name
        self.is_white = is_white
        self.wins = wins
        self.losses = losses
        self.draws = draws

    def __str__(self):
        return self.name

    def next_move(self, board):
        moves = list(board.legal_moves)
        move_idx = np.random.randint(len(moves))
        return moves[move_idx]

    def get_stats(self):
        total_games = self.wins + self.losses + self.draws
        win_perc = self.wins / total_games * 100
        loss_perc = self.losses / total_games * 100
        draw_perc = self.draws / total_games * 100
        return "{}: {} ({:.1f}%) - {} ({:.1f}%) - {} ({:.1f}%)".format(
            self.name, self.wins, win_perc, self.losses, loss_perc, self.draws,
            draw_perc)
