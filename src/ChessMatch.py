import chess
import random


class ChessMatch:
    def __init__(self, player_1, player_2):
        self.board = chess.Board()
        self.players = [player_1, player_2]

    def __str__(self):
        return str(self.board)

    def randomize_sides(self):
        random.shuffle(self.players)

    def play_game(self):
        move_idx = 0
        print(self)
        while not self.board.is_game_over():
            print(
                self.get_color_from_move(move_idx) + ":",
                self.players[move_idx % 2], "to move")
            move = self.players[move_idx % 2].next_move(self.board)
            self.board.push(move)
            print(self)
            move_idx += 1
        print("Game result:", self.board.result())

    def get_color_from_move(self, move_idx):
        if move_idx == 0:
            return 'White'
        else:
            return 'Black'
