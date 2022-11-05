from Player import Player
import keras
from keras.layers import Input, Dense, Embedding, Flatten, Reshape, concatenate, Conv2D, BatchNormalization, Dropout, Add
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import chess
import glob


class TheGambit(Player):
    def __init__(self,
                 name='TheGambit',
                 wins=0,
                 losses=0,
                 draws=0,
                 load_model=False,
                 noise_level=0.05,
                 min_noise_level=1e-3):
        Player.__init__(self, name=name, wins=wins, losses=losses, draws=draws)
        self.noise_level = noise_level
        self.min_noise_level = min_noise_level
        self.games_per_version = 32
        self.load_model = load_model
        self.model = self.get_model()
        self.color = True
        self.min_max_func = np.argmax
        self.training = True
        self.setup_training_data()
        self.model_version = 0
        self.total_games_trained = 0
        self.next_save_count = 2**10
        self.val_mae = []
        self.train_mae = []
        self.moves_per_game = []

    def setup_training_data(self):
        board = chess.Board()
        moves = list(board.legal_moves)
        board_grid_list = []
        board_stat_list = []
        for move in moves:
            board.push(move)
            grid, stats = self.board_to_training(board)
            board_grid_list.append(grid)
            board_stat_list.append(stats)
            board.pop()

        self.x_val = [np.array(board_grid_list), np.array(board_stat_list)]
        self.y_val = np.zeros(len(self.x_val[0])) + 0.5

    def set_color(self, color):
        """
        color: chess.WHITE or chess.BLACK
        """
        self.color = color
        if color == chess.WHITE:
            self.min_max_func = np.argmax
        else:
            self.min_max_func = np.argmin

    def get_model(self):
        if self.load_model:
            model_paths = glob.glob(f'models/{self.name}_v*')
            if len(model_paths) > 0:
                newest_model_path = ''
                newest_model_version = -1
                for path in model_paths:
                    version = int(path.strip().split('_v')[-1])
                    if version > newest_model_version:
                        newest_model_path = path
                        newest_model_version = version
                print('loading:', newest_model_path)
                self.model_version = newest_model_version + 1
                for idx in range(self.model_version):
                    self.adjust_noise_level(self.games_per_version)
                print(self.noise_level)
                return keras.models.load_model(path)
            else:
                print('No save models found. Starting from scratch')

        # create model from scratch
        max_chess_piece_idx = 12
        embedding_size = 4
        cnn_filter_power = 5
        cnn_filter_shape = (3, 3)
        activation = "elu"
        dense_power = 6
        cnn_stack_size = 3
        mlp_stack_size = 3
        dropout_rate = 0.2

        # input for the 64 square chess board (embedding)
        board_input = Input(shape=[8, 8], name='board_input')
        # input for
        state_input = Input(shape=[
            6,
        ], name='state_input')

        flat_board = Flatten(name='flat_board')(board_input)
        embeddings = Embedding(max_chess_piece_idx + 1,
                               embedding_size,
                               name='embed_board')(flat_board)
        cnn = Reshape((8, 8, embedding_size), name='cnn_board')(embeddings)
        for idx in range(cnn_stack_size):
            cnn = Conv2D(2**(cnn_filter_power + idx),
                         cnn_filter_shape,
                         activation=activation,
                         name=f'cnn_{idx}')(cnn)
        flatten_board_cnn = Flatten(name=f'flatten_cnn')(cnn)

        elu_stack = concatenate((flatten_board_cnn, state_input),
                                name='cnn_state_merge')
        for idx in range(mlp_stack_size):
            elu_stack = BatchNormalization(name=f'batch_norm_{idx}')(elu_stack)
            elu_stack = Dense(2**(dense_power + mlp_stack_size - idx - 1),
                              activation=activation,
                              name=f'dense_{idx}')(elu_stack)
            elu_stack = Dropout(dropout_rate, name=f'dropout_{idx}')(elu_stack)
        output = Dense(1, activation="sigmoid", name=f'output')(elu_stack)

        model = Model(inputs=[board_input, state_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=2e-4), loss='mae')
        model.summary()
        return model

    def save_model(self):
        path = f'models/{self.name}_v{self.model_version}'
        self.model.save(path)
        self.model_version += 1

    def board_to_grid(self, board):
        np_board = np.zeros((8, 8), dtype=int)
        for col_idx in range(8)[::-1]:
            for row_idx, letter in enumerate(
                ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')):
                square = eval(f"chess.{letter}{col_idx+1}")
                piece = board.piece_at(square)
                if piece is not None:
                    piece_type = piece.piece_type
                    piece_color = piece.color
                    value = piece_type * 2 - piece_color
                    np_board[7 - col_idx, row_idx] = value
        return np_board

    def get_board_stats(self, board):
        stats = np.zeros([6], dtype=float)
        stats[0] = board.has_kingside_castling_rights(chess.WHITE)
        stats[1] = board.has_queenside_castling_rights(chess.WHITE)
        stats[2] = board.has_kingside_castling_rights(chess.BLACK)
        stats[3] = board.has_queenside_castling_rights(chess.BLACK)
        stats[4] = board.halfmove_clock / 50
        stats[4] = int(board.turn)
        return stats

    def board_to_training(self, board):
        return self.board_to_grid(board), self.get_board_stats(board)

    def next_move(self, board):
        self.set_color(board.turn)
        moves = list(board.legal_moves)
        board_grid_list = []
        board_stat_list = []
        for move in moves:
            board.push(move)
            grid, stats = self.board_to_training(board)
            board_grid_list.append(grid)
            board_stat_list.append(stats)
            board.pop()

        predictions = self.model.predict(
            [np.array(board_grid_list),
             np.array(board_stat_list)], verbose=0).flatten()
        # add noise for better training data
        noise = np.random.uniform(0, 1, len(predictions)) * self.noise_level
        move_idx = self.min_max_func(predictions + noise)
        return moves[move_idx]

    def get_move_prompt(self, board):
        string = "Choose move from list: "
        for move in board.legal_moves:
            string += str(move) + " "
        string += "\n"
        return string

    def adjust_noise_level(self, num_games):
        num_base_games = 512
        self.noise_level *= max(0.1,
                                (num_base_games - num_games) / num_base_games)
        self.noise_level = max(self.noise_level, self.min_noise_level)

    def fit_games(self, training_set, num_games):
        num_boards = len(training_set)
        self.total_games_trained += num_games
        print(
            f'{self.name} is training on {num_boards} boards ({num_boards/num_games/2} moves per game)'
        )

        x_train_grid = []
        x_train_stats = []
        y_train = []
        counter = 0
        for board, label in training_set:
            grid, stats = self.board_to_training(board)
            x_train_grid.append(grid)
            x_train_stats.append(stats)
            y_train.append(float(label))
            counter += 1
        y_train = np.array(y_train)
        x_train = [np.array(x_train_grid), np.array(x_train_stats)]

        # guess average
        pred_val = y_train.mean()
        mae = np.mean(np.abs(y_train - pred_val))
        mae_val = np.mean(np.abs(self.y_val - pred_val))
        print(f'Naive MAE: {mae:.4f} (val: {mae_val:.4f})')

        # train model
        self.model.fit(x_train,
                       y_train,
                       validation_data=(self.x_val, self.y_val),
                       batch_size=1024,
                       epochs=8)

        # log training history
        pred_val = self.model.predict(x_train)
        mae = np.sqrt(np.mean((y_train - pred_val)**2))
        pred_val = self.model.predict(self.x_val)
        mae_val = np.sqrt(np.mean((self.y_val - pred_val)**2))
        print(f'train mae: {mae}\nval mae: {mae_val}')
        self.train_mae.append(mae)
        self.val_mae.append(mae_val)
        self.moves_per_game.append(num_boards / num_games / 2)

        plt.plot(range(1,
                       len(self.train_mae[1:]) + 1),
                 self.train_mae[1:],
                 label='Training')
        plt.plot(range(len(self.val_mae[1:])),
                 self.val_mae[1:],
                 label='Validation')
        plt.legend()
        plt.xlabel('trainings')
        plt.ylabel('MAE')
        plt.savefig(f'graphs/{self.name}_training_mae.png')
        plt.clf()

        plt.plot(self.moves_per_game[1:])
        plt.xlabel('trainings')
        plt.ylabel('Moves per game')
        plt.savefig(f'graphs/{self.name}_moves_per_game.png')
        plt.clf()

        # update validation
        self.x_val = [np.array(x_train_grid), np.array(x_train_stats)]
        self.y_val = y_train

        # save model
        if self.total_games_trained > self.next_save_count:
            self.next_save_count += 2**10
            self.save_model()

        # reduce noise
        self.adjust_noise_level(num_games)
        print('Noise level:', self.noise_level)