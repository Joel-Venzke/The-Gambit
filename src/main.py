from ChessMatch import ChessMatch
from Player import Player

p1 = Player(name='Player 1')
p2 = Player(name='Player 2')

chess_match = ChessMatch(p1, p2)
chess_match.randomize_sides()
chess_match.play_game()
