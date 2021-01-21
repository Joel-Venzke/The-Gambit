from ChessMatch import ChessMatch
from Player import Player
from Human import Human
from Stockfish import Stockfish
import time

num_games = 100

p1 = Player(name='Player 1')
p2 = Stockfish(time=1e-6)
# p2 = Player(name='Player 2')
# p2 = Human()

chess_match = ChessMatch(p1, p2, verbose=1)
chess_match.randomize_sides()
start = time.time()
for idx in range(num_games):
    chess_match.play_game()
    chess_match.randomize_sides()
avg_game_time = (time.time() - start) / num_games
chess_match.player_stats()
chess_match.quit()
print("Avg game time:", avg_game_time)
print('Games per hour:', 3600 / avg_game_time)
