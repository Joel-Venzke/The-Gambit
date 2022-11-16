from ChessMatch import ChessMatch
from Player import Player
from Human import Human
from StockfishPlayer import StockfishPlayer
from TheGambit import TheGambit
import time

# player_1 = Player(name='Player 1')
# human = Human()
the_gambit_1 = TheGambit(name='TheGambitZero_0',
                         load_model=True,
                         noise_level=0.0,
                         min_noise_level=0)
# player_2 = StockfishPlayer()
player_2 = TheGambit(name='TheGambitZero_1',
                     load_model=True,
                     noise_level=0.0,
                     min_noise_level=0)
# player_2 = Player(name='Random Player')

num_games = 2**17
for depth in range(1):
    chess_match = ChessMatch(the_gambit_1,
                             player_2,
                             training_batch_size=64,
                             verbose=1)
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
