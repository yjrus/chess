import chess

EXIT = -101
NEW_GAME = -102
CANCEL_MOVE = -103
HISTORY = -104

WHITE_WIN = -200
BLACK_WIN = -201
DRAW = -202


commands = {"!exit", "!new", "!giveup", "!cmove", "!draw", "!hist", "!see"}

class Chess(chess.Board):
    def __init__(self):
        super().__init__()
        self.fen_log = dict()
        self.fen_log[0] = self.fen()
        pass

    def push_(self, move: chess.Move) -> None:
        self.push(move)
        self.fen_log[len(self.move_stack)] = self.fen()

    def reset_(self) -> None:
        self.fen_log.clear()
        self.reset()
        
    def print_(self, fen_str: str = None) -> None:
        if fen_str is None:
            unicode_line = self.unicode(borders = False, empty_square = ". ")
        else:
            tmp_board = chess.Board(fen_str)
            unicode_line = tmp_board.unicode(borders = False, empty_square = ". ")
            del tmp_board

        lines = unicode_line.split("\n")
        
        print("  A  B  C  D  E  F  G  H")
        
        row = 8
        for line in lines:
            print(f"{row} {line}")
            row -= 1

    def get_moves_str(self) -> str:
        hist = ""
        for i, move in enumerate(self.move_stack):
            hist += f"{str(i//2+1)+") " if i%2==0 else " "*(len(str((i-1)//2+1))+2)}"
            hist += f"{move.uci()}\n"
        return hist

    def get_fen(self, N: int = -1) -> str:
        if N == -1:
            return self.fen()
        
        if 0 <= N <= len(self.move_stack):
            print("Searching...")
            return self.fen_log.get(N, "Error")
        
        return "Error"


def user_input(board: Chess, gameType = "uu") -> int:
    in_ = input("Please, enter uci move or command: ").lower()
    if len(in_) >= 1:
        in_ = list(map(str, in_.split()))
    else:
        return -1
        
    if in_[0].lower() in commands:
        cmd = in_[0]
        if len(in_) > 1:
            arg = int(in_[1])
        
        if cmd == "!exit":
            return EXIT
            
        elif cmd == "!new":
            return NEW_GAME
            
        elif cmd == "!giveup":
            if board.turn == chess.WHITE:
                print("Whites gave up")
                return BLACK_WIN
            else:
                print("Blacks gave up")    
                return WHITE_WIN
                
        elif cmd == "!draw":
            if game_type == "uu":
                answ = input("Does the opponent agree? (y/n): ").lower()
                if answ == "y":
                    return DRAW
                else:
                    return -1
                    
            return DRAW

        elif cmd == "!cmove":
            board.pop()
            return CANCEL_MOVE

        elif cmd == "!hist":
            print(board.get_moves_str())
            return HISTORY
            
        elif cmd == "!see":
            if arg > len(board.move_stack) or arg < 0:
                return -1
            else:
                fen = board.get_fen(arg)
                if fen == "Error":
                    print(fen)
                    return -1
                board.print_(fen)
                return HISTORY
                    
    else:
        try:
            move = chess.Move.from_uci(in_[0])
        except ValueError as e:
            print(e)
            return -1
            
        if move in board.legal_moves:
            board.push_(move)
            return 0
        else:
            return -1
        


def uu_game(board: Chess) -> int:
    game_on = True
    board = Chess()
    game_result = DRAW

    while game_on:
        board.print_()
        input_ = -1
        while input_ == -1:
            input_ = user_input(board)
            if input_ == -1:
                print("Try again")
            elif input_ in {DRAW, WHITE_WIN, BLACK_WIN, EXIT, NEW_GAME}:
                return input_
            elif input_ == HISTORY:
                input_ = -1
                input("Введите любой символ, что бы продолжить: ")
                board.print_()
            elif input_ == CANCEL_MOVE:
                board.print_()
                input_ = -1     

        if board.is_game_over():
            if board.is_checkmate():
                if board.turn == chess.WHITE:
                    game_result = BLACK_WIN
                else:
                    game_result = WHITE_WIN
            board.print_()
            game_on = False
            
        elif board.is_fifty_moves() or board.is_repetition(3):
            board.print_()
            game_on = False
                
    print("Game was ended")
    return game_result


if __name__ == "__main__":
    print("Commands:")
    print("!exit - exit")
    print("!new - new game")
    print("!giveup - give up")
    print("!сmove - cancel the move")
    print("!draw - draw")
    print("!hist - view the history of moves")
    print("!see N - view the state of the game on the N turn")

    board = Chess()
    while True:
        print("-==-"*20)
        current_game = uu_game(board)
        if current_game == EXIT:
            break

        if current_game == WHITE_WIN:
            print("White win:  1  -  0  ")
        elif current_game == BLACK_WIN:
            print("Black win:  0  -  1  ")
        elif current_game == DRAW:
            print("Draw: 1/2 - 1/2")

        if input("New game? (y/n): ").lower() == "n":
            break

        board.reset_()
        
    print("Goodbye!")
        
        
            

    