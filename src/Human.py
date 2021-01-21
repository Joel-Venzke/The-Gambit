from Player import Player
import chess


class Human(Player):
    def __init__(self, is_white=True, name='Human', wins=0, losses=0, draws=0):
        Player.__init__(self,
                        is_white=is_white,
                        name=name,
                        wins=wins,
                        losses=losses,
                        draws=draws)

    def next_move(self, board):
        prompt = self.get_move_prompt(board)
        move = None
        make_move = True
        while make_move:
            move = input(prompt)
            try:
                move = chess.Move.from_uci(move)
                if move in board.legal_moves:
                    make_move = False
                else:
                    print(move, "is not a legal move")
            except ValueError:
                print("Bad move, try again")

        return move

    def get_move_prompt(self, board):
        string = "Choose move from list: "
        for move in board.legal_moves:
            string += str(move) + " "
        string += "\n"
        return string
