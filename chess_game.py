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
    """
    Расширенный класс шахматной доски с дополнительным функционалом
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.fen_log = dict()
        self.fen_log[0] = self.fen()



    def push_(self, move: chess.Move) -> None:
        """
        Делает ход и сохраняет FEN в лог
        """
        self.push(move)
        self.fen_log[len(self.move_stack)] = self.fen()



    def reset_(self) -> None:
        """
        Сбрасывает доску и очищает лог FEN
        """
        self.fen_log.clear()
        self.reset()



    def print_(self, fen_str: str = None) -> None:
        """
        Печатает доску в консоль
        """
        if fen_str is None:
            unicode_line = self.unicode(borders=False, empty_square=".")
        else:
            tmp_board = chess.Board(fen_str)
            unicode_line = tmp_board.unicode(borders=False, empty_square=".")
            del tmp_board
        
        lines = unicode_line.split("\n")
        
        print("  A B C D E F G H")
        
        row = 8
        for line in lines:
            print(f"{row} {line}")
            row -= 1



    def get_moves_str(self) -> str:
        """
        Возвращает строку с историей ходов в UCI формате
        """
        hist = ""
        
        for i, move in enumerate(self.move_stack):
            prefix = f"{str(i//2+1)+') '}" if i % 2 == 0 else ' ' * (len(str((i-1)//2+1)) + 2)
            hist += f"{prefix}{move.uci()}\n"
        
        return hist



    def get_fen(self, n: int = -1) -> str:
        """
        Возвращает FEN для заданного номера полухода
        """
        if n == -1:
            return self.fen()
        
        if 0 <= n <= len(self.move_stack):
            print("Searching...")
            return self.fen_log.get(n, "Error")
        
        return "Error"



def userInput(board: Chess, game_type: str = "uu") -> int:
    """
    Обрабатывает пользовательский ввод (ходы или команды)
    """
    input_str = input("Please, enter uci move or command: ").lower()
    
    if len(input_str) >= 1:
        input_parts = list(map(str, input_str.split()))
    else:
        return -1
    
    if input_parts[0].lower() in commands:
        cmd = input_parts[0]
        
        if len(input_parts) > 1:
            arg = int(input_parts[1])
        
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
                answer = input("Does the opponent agree? (y/n): ").lower()
                if answer == "y":
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
            move = chess.Move.from_uci(input_parts[0])
        except ValueError as e:
            print(e)
            return -1
        
        if move in board.legal_moves:
            board.push_(move)
            return 0
        else:
            return -1



def uuGame(board: Chess) -> int:
    """
    Запускает игру пользователь против пользователя
    """
    game_on = True
    board = Chess()
    game_result = DRAW
    
    while game_on:
        board.print_()
        user_input_result = -1
        
        while user_input_result == -1:
            user_input_result = userInput(board)
            
            if user_input_result == -1:
                print("Try again")
            
            elif user_input_result in {DRAW, WHITE_WIN, BLACK_WIN, EXIT, NEW_GAME}:
                return user_input_result
            
            elif user_input_result == HISTORY:
                user_input_result = -1
                input("Press any key to continue: ")
                board.print_()
            
            elif user_input_result == CANCEL_MOVE:
                board.print_()
                user_input_result = -1
        
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
    print("!cmove - cancel the move")
    print("!draw - draw")
    print("!hist - view the history of moves")
    print("!see N - view the state of the game on the N turn")
    
    board = Chess()
    
    while True:
        print("-==-" * 20)
        current_game = uuGame(board)
        
        if current_game == EXIT:
            break
        
        if current_game == WHITE_WIN:
            print("White win:  1  -  0")
        elif current_game == BLACK_WIN:
            print("Black win:  0  -  1")
        elif current_game == DRAW:
            print("Draw: 1/2 - 1/2")
        
        if input("New game? (y/n): ").lower() == "n":
            break
        
        board.reset_()
    
    print("Goodbye!")
