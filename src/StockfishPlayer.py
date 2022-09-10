from Player import Player
from stockfish import Stockfish
import chess
import chess.engine


class StockfishPlayer(Player):
    def __init__(
            self,
            name='Stockfish',
            wins=0,
            losses=0,
            draws=0,
            engine_path="/Users/joelvenzke/Repos/The-Gambit/venv/lib/python3.7/site-packages/stockfish/models.py",
            time=None,
            depth=None):
        Player.__init__(self, name=name, wins=wins, losses=losses, draws=draws)
        self.engine = Stockfish()
        # self.depth = depth
        # self.time = time

    def next_move(self, board):
        print(board)
        exit()
        result = self.engine.play(
            board, chess.engine.Limit(time=self.time, depth=self.depth))
        return result.move

    def quit(self):
        self.engine.quit()
