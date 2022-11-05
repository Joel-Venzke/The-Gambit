import chess
import random
import chess.svg


class ChessMatch:
    def __init__(self, player_1, player_2, training_batch_size=32, verbose=2):
        self.board = chess.Board()
        self.players_fixed_order = [player_1, player_2]
        self.players = [player_1, player_2]
        self.num_training = player_1.training + player_2.training
        self.verbose = verbose
        self.training_batch_size = training_batch_size
        self.game_counter = 0
        self.reset_training_set()

    def __str__(self):
        return str(self.board)

    def reset_training_set(self):
        self.non_draw_game = 0
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
            if self.verbose == 2 or (self.verbose == 1 and self.game_counter %
                                     self.training_batch_size == 0):
                print(self.board.fen())
            if self.verbose >= 3:
                print(self)
            move_idx += 1
        # print result
        if self.verbose == 2 or (self.verbose == 1 and self.game_counter %
                                 self.training_batch_size == 0):
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
        save_game = False
        if self.training_result == 1.0 or self.training_result == 0.0:
            save_game = True
            self.non_draw_game += 1
        elif self.training_result == 0.5 and self.non_draw_game > 0:
            self.non_draw_game -= 1
            save_game = True
        if save_game:
            num_moves = len(self.game_log)
            for move_idx, board in enumerate(self.game_log):
                # scale board rating by closeness to end of game
                # self.training_set.append([
                #     board, self.training_result -
                #     ((self.training_result - 0.5) *
                #      (num_moves - move_idx - 1) / num_moves)
                # ])
                self.training_set.append([board, self.training_result])

    def run_training(self):
        if len(self.training_set) > 0:
            if self.num_training == 1:
                self.player_stats()
                for player in self.players:
                    if player.training:
                        player.fit_games(self.training_set,
                                         self.training_batch_size)
            if self.num_training == 2:
                self.player_stats()
                self.players_fixed_order[(
                    (self.game_counter // self.training_batch_size) + 1) %
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
