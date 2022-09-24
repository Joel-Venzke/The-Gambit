from ChessMatch import ChessMatch
from Player import Player
from Human import Human
from StockfishPlayer import StockfishPlayer
from TheGambit import TheGambit
import time

player_1 = Player(name='Player 1')
player_2 = Player(name='Player 2')
human = Human()
stockfish = StockfishPlayer()
the_gambit = TheGambit(name='TheGambit')
the_gambit_1 = TheGambit(name='TheGambit 1')
the_gambit_2 = TheGambit(name='TheGambit 2')

num_games = 2**10
for depth in range(1):
    chess_match = ChessMatch(the_gambit_1,
                             the_gambit_2,
                             training_batch_size=32,
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
