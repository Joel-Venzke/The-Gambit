from ChessMatch import ChessMatch
from Player import Player
from Human import Human
from StockfishPlayer import StockfishPlayer
import time

player_1 = Player(name='Player 1')
player_2 = Player(name='Player 2')
human = Human()
stockfish = StockfishPlayer()

num_games = 5
for depth in range(1):
    chess_match = ChessMatch(player_1, Stockfish, verbose=2)
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
