from Player import Player
import keras
from keras.layers import Input, Dense, Embedding, Flatten, Reshape, concatenate, Conv2D, BatchNormalization, Dropout, Add
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import chess
import glob


class BoardLoader(keras.utils.Sequence):
    def __init__(self, fen_list, labels, batch_size, base_file_path):
        self.fen_list = fen_list
        self.labels = labels
        self.batch_size = batch_size
        self.base_file_path = base_file_path

    def __len__(self):
        return (np.ceil(len(self.fen_list) / float(self.batch_size))).astype(
            np.int)

    def board_to_file(self, fen):
        return fen.replace('/', '_')

    def get_board_data_files(self, fen):
        grid_file = f'{self.base_file_path}{self.board_to_file(fen)}_grid.npy'
        stats_file = f'{self.base_file_path}{self.board_to_file(fen)}_stats.npy'
        return grid_file, stats_file

    def load_board(self, fen):
        grid_file, stats_file = self.get_board_data_files(fen)
        grid = np.load(grid_file)
        stats = np.load(stats_file)
        return grid, stats

    def __getitem__(self, idx):
        batch_x = self.fen_list[idx * self.batch_size:(idx + 1) *
                                self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) *
                              self.batch_size]
        grid_list = []
        stats_list = []
        for fen in batch_x:
            grid, stats = self.load_board(fen)
            grid_list.append(grid)
            stats_list.append(stats)
        data = [np.array(grid_list), np.array(stats_list)], np.array(batch_y)
        return data


class TheGambit(Player):
    def __init__(self,
                 name='TheGambit',
                 wins=0,
                 losses=0,
                 draws=0,
                 load_model=False,
                 noise_level=0.00,
                 min_noise_level=0):
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
        self.batch_size = 1024
        self.val_mae = []
        self.train_mae = []
        self.moves_per_game = []
        self.data_name = f'train_data/{name}.csv'
        self.board_data_name = 'boards/'

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
            5,
        ], name='state_input')

        flat_board = Flatten(name='flat_board')(board_input)
        embeddings = Embedding(max_chess_piece_idx + 1,
                               embedding_size,
                               name='embed_board')(flat_board)
        cnn_board = Reshape((8, 8, embedding_size),
                            name='cnn_board')(embeddings)
        cnn = Conv2D(2**(cnn_filter_power),
                     cnn_filter_shape,
                     activation=activation,
                     name=f'cnn_{0}')(cnn_board)
        for idx in range(1, cnn_stack_size):
            cnn = Conv2D(2**(cnn_filter_power + idx),
                         cnn_filter_shape,
                         activation=activation,
                         name=f'cnn_{idx}')(cnn)
        wide_cnn = Conv2D(2**(cnn_filter_power + cnn_stack_size), (8, 8),
                          activation=activation,
                          name=f'wide_cnn')(cnn_board)
        flatten_board_cnn = Flatten(name=f'flatten_cnn')(cnn)
        flatten_board_wide_cnn = Flatten(name=f'flatten_wide_cnn')(wide_cnn)

        elu_stack = concatenate(
            (flatten_board_wide_cnn, flatten_board_cnn, state_input),
            name='cnn_state_merge')
        for idx in range(mlp_stack_size):
            elu_stack = Dense(2**(dense_power + mlp_stack_size - idx - 1),
                              activation=activation,
                              name=f'dense_{idx}')(elu_stack)
            elu_stack = BatchNormalization(name=f'batch_norm_{idx}')(elu_stack)
            elu_stack = Dropout(dropout_rate, name=f'dropout_{idx}')(elu_stack)
        post_elu = concatenate((elu_stack, state_input), name='final_merge')
        output = Dense(1, activation="linear", name=f'output')(post_elu)

        model = Model(inputs=[board_input, state_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=1e-5),
                      loss='mean_squared_error')
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
        stats = np.zeros([5], dtype=float)
        stats[0] = board.has_kingside_castling_rights(chess.WHITE)
        stats[1] = board.has_queenside_castling_rights(chess.WHITE)
        stats[2] = board.has_kingside_castling_rights(chess.BLACK)
        stats[3] = board.has_queenside_castling_rights(chess.BLACK)
        stats[4] = board.halfmove_clock / 50
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

    def board_to_file(self, fen):
        return fen.replace('/', '_')

    def get_board_data_files(self, fen):
        grid_file = f'{self.board_data_name}{self.board_to_file(fen)}_grid.npy'
        stats_file = f'{self.board_data_name}{self.board_to_file(fen)}_stats.npy'
        return grid_file, stats_file

    def save_board(self, board):
        grid, stats = self.board_to_training(board)
        grid_file, stats_file = self.get_board_data_files(board.fen())
        np.save(grid_file, grid)
        np.save(stats_file, stats)

    def load_board(self, fen):
        grid_file, stats_file = self.get_board_data_files(fen)
        grid = np.load(grid_file)
        stats = np.load(stats_file)
        return grid, stats

    def fit_games(self, training_set, num_games):
        num_boards = len(training_set)
        self.total_games_trained += num_games

        records = []
        for board, label in training_set:
            records.append({'board': board.fen(), 'label': float(label)})
            self.save_board(board)
        new_training_df = pd.DataFrame(records)
        # save new boards to records
        new_training_df.to_csv(self.data_name,
                               mode='a',
                               index=False,
                               header=False)
        # load all boards played
        training_df = pd.read_csv(self.data_name,
                                  names=new_training_df.columns,
                                  header=None)
        print('Total boards in database:', len(training_df))
        # get average result for all boards
        training_df = training_df.groupby('board').mean().reset_index()

        print(f'{self.name} is training on {len(training_df)} boards')

        train_df, val_df = train_test_split(training_df, test_size=0.2)
        train_set = BoardLoader(np.array(list(train_df['board'].values)),
                                np.array(list(train_df['label'].values)),
                                self.batch_size, self.board_data_name)

        val_set = BoardLoader(np.array(list(val_df['board'].values)),
                              np.array(list(val_df['label'].values)),
                              self.batch_size, self.board_data_name)

        # train model
        callbacks = [
            EarlyStopping(monitor='val_loss',
                          patience=30,
                          restore_best_weights=True)
        ]
        self.model.fit(train_set,
                       validation_data=val_set,
                       batch_size=self.batch_size,
                       epochs=1,
                       callbacks=callbacks)

        # log training history
        pred = self.model.predict(train_set)
        rmse = np.sqrt(mean_squared_error(train_set.labels, pred))
        pred_val = self.model.predict(val_set)
        rmse_val = np.sqrt(mean_squared_error(val_set.labels, pred_val))
        print(f'train rmse: {rmse}\nval rmse: {rmse_val}')
        self.train_mae.append(rmse)
        self.val_mae.append(rmse_val)
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
        plt.ylabel('RMSE')
        plt.savefig(f'graphs/{self.name}_training_mae.png')
        plt.clf()

        plt.plot(self.moves_per_game[1:])
        plt.xlabel('trainings')
        plt.ylabel('Moves per game')
        plt.savefig(f'graphs/{self.name}_moves_per_game.png')
        plt.clf()

        # save model
        if self.total_games_trained > self.next_save_count:
            self.next_save_count += 2**10
            self.save_model()

        # reduce noise
        self.adjust_noise_level(num_games)
        print('Noise level:', self.noise_level)
