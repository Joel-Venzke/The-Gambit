from ChessMatch import ChessMatch
from Player import Player
from Human import Human
import time

num_games = 1

p1 = Player(name='Player 1')
# p2 = Player(name='Player 2')
p2 = Human()

chess_match = ChessMatch(p1, p2, verbose=2)
chess_match.randomize_sides()
start = time.time()
for idx in range(num_games):
    chess_match.play_game()
    chess_match.randomize_sides()
avg_game_time = (time.time() - start) / num_games
chess_match.player_stats()
print("Avg game time:", avg_game_time)
print('Games per hour:', 3600 / avg_game_time)
