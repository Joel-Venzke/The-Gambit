import chess
import random
import chess.svg


class ChessMatch:
    def __init__(self, player_1, player_2, verbose=2):
        self.board = chess.Board()
        self.players = [player_1, player_2]
        self.verbose = verbose

    def __str__(self):
        return str(self.board)

    def randomize_sides(self):
        random.shuffle(self.players)

    def play_game(self):
        # reset board
        self.reset_board()
        move_idx = 0
        if self.verbose >= 2:
            print(self)

        # play game
        while not self.board.is_game_over():
            if self.verbose >= 2:
                print(
                    self.get_color_from_move(move_idx) + ":",
                    self.players[move_idx % 2], "to move")
            # make move
            move = self.players[move_idx % 2].next_move(self.board)
            self.board.push(move)
            if self.verbose >= 2:
                print(self)
            move_idx += 1
        # print result
        self.report_result()

    def get_color_from_move(self, move_idx):
        if move_idx == 0:
            return 'White'
        else:
            return 'Black'

    def reset_board(self):
        self.board = chess.Board()

    def report_result(self):
        result = self.board.result()
        if self.verbose >= 1:
            print("Game result:", result,
                  "({}, {})".format(self.players[0], self.players[1]))
        if result == '1-0':
            self.players[0].wins += 1
            self.players[1].losses += 1
        elif result == '0-1':
            self.players[0].losses += 1
            self.players[1].wins += 1
        elif result == '1/2-1/2':
            self.players[0].draws += 1
            self.players[1].draws += 1

    def player_stats(self):
        print("win (%) - loss (%) - draws (%)")
        for player in self.players:
            print(player.get_stats())
