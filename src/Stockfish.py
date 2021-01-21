from Player import Player
import chess
import chess.engine


class Stockfish(Player):
    def __init__(self,
                 is_white=True,
                 name='Stockfish',
                 wins=0,
                 losses=0,
                 draws=0,
                 engine_path="/usr/local/Cellar/stockfish/12/bin/stockfish",
                 time=0.01):
        Player.__init__(self,
                        is_white=is_white,
                        name=name,
                        wins=wins,
                        losses=losses,
                        draws=draws)
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.time = time

    def next_move(self, board):
        result = self.engine.play(board, chess.engine.Limit(time=self.time))
        return result.move

    def quit(self):
        self.engine.quit()
