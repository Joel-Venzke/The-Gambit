import chess
import random
import chess.svg


class ChessMatch:
    def __init__(self, player_1, player_2, training_batch_size=32, verbose=2):
        self.board = chess.Board()
        self.players = [player_1, player_2]
        self.num_training = player_1.training + player_2.training
        self.verbose = verbose
        self.training_batch_size = training_batch_size
        self.game_counter = 0
        self.reset_training_set()

    def __str__(self):
        return str(self.board)

    def reset_training_set(self):
        self.training_set = []

    def randomize_sides(self):
        random.shuffle(self.players)

    def play_game(self):
        self.game_counter += 1
        # reset board
        self.reset_board()
        move_idx = 0
        if self.verbose >= 3:
            print(self)

        self.game_log = []

        # play game
        while not self.board.is_game_over():
            self.game_log.append(self.board.copy(stack=False))
            if self.verbose >= 3:
                print(
                    self.get_color_from_move(move_idx) + ":",
                    self.players[move_idx % 2], "to move")
            # make move
            move = self.players[move_idx % 2].next_move(self.board)
            self.board.push(move)
            if self.verbose == 2:
                if move_idx % 2 == 0:
                    print(f'{move_idx//2 + 1}.{move}', end='')
                else:
                    print(f',{move} ', end='')
            if self.verbose >= 3:
                print(self)
            move_idx += 1
        # print result
        if self.verbose == 2:
            print()
        self.report_result(move_idx)
        if self.game_counter % self.training_batch_size == 0:
            self.run_training()

    def get_color_from_move(self, move_idx):
        if move_idx % 2 == 0:
            return 'White'
        else:
            return 'Black'

    def reset_board(self):
        self.board = chess.Board()

    def report_result(self, move_idx):
        result = self.board.result()
        if self.verbose >= 1:
            print(
                f"Game {self.game_counter} result: {result} ({self.players[0]}, {self.players[1]}) in {move_idx/2} moves"
            )
        if result == '1-0':
            self.players[0].wins += 1
            self.players[1].losses += 1
            self.training_result = 1.0
        elif result == '0-1':
            self.players[0].losses += 1
            self.players[1].wins += 1
            self.training_result = 0.0
        elif result == '1/2-1/2':
            self.players[0].draws += 1
            self.players[1].draws += 1
            self.training_result = 0.5
        self.gen_training_data()

    def gen_training_data(self):
        for board in self.game_log:
            self.training_set.append([board, self.training_result])

    def run_training(self):
        if self.num_training == 1:
            for player in self.players:
                if player.training:
                    player.fit_games(self.training_set,
                                     self.training_batch_size)
        if self.num_training == 2:
            self.players[(self.game_counter // self.training_batch_size) %
                         2].fit_games(self.training_set,
                                      self.training_batch_size)
        self.reset_training_set()

    def player_stats(self):
        print("\nname: win (%) - loss (%) - draws (%)")
        for player in self.players:
            print(player.get_stats())
        print()

    def quit(self):
        for player in self.players:
            player.quit()
