import torch
import chess
from chess_game import Chess, EXIT, NEW_GAME, CANCEL_MOVE, HISTORY, WHITE_WIN, BLACK_WIN, DRAW, userInput
from model import ChessModel
from chess_data_processing import MCTS, MCTS_simulations, boardToTensor, selectMoveWithTemperature



def botMove(board: Chess,
            model: ChessModel,
            simulations: int = 100,
            temperature: float = 0.5) -> chess.Move:
    """
    Вычисляет ход бота с помощью MCTS
    """
    root = MCTS(move=None)
    move = MCTS_simulations(root, model, board, count=simulations)
    
    if move is None:
        return None
    
    return move



def ubGame(board: Chess,
           model: ChessModel,
           bot_plays_white: bool = False,
           simulations: int = 100,
           temperature: float = 0.5) -> int:
    """
    Запускает игру пользователь против бота
    
    Args:
        board: доска Chess
        model: обученная модель
        bot_plays_white: True если бот играет белыми
        simulations: количество симуляций MCTS
        temperature: температура для выбора хода
    
    Returns:
        код результата игры
    """
    game_on = True
    game_result = DRAW
    
    print("\n" + "=" * 50)
    print("USER vs BOT GAME")
    if bot_plays_white:
        print("Bot plays WHITE, User plays BLACK")
    else:
        print("User plays WHITE, Bot plays BLACK")
    print("=" * 50 + "\n")
    
    while game_on:
        board.print_()
        
        # ХОД ПОЛЬЗОВАТЕЛЯ
        if board.turn == (chess.BLACK if bot_plays_white else chess.WHITE):
            print("\nYour turn...")
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
        
        # ХОД БОТА
        else:
            
            move = botMove(
                board=board,
                model=model,
                simulations=simulations,
                temperature=temperature
            )
            
            if move is None:
                print("Bot has no legal moves")
                break
            
            print(f"Bot plays: {move.uci()}")
            board.push_(move)
        
        # ПРОВЕРКА ОКОНЧАНИЯ ИГРЫ
        if board.is_game_over():
            if board.is_checkmate():
                if board.turn == chess.WHITE:
                    game_result = BLACK_WIN
                else:
                    game_result = WHITE_WIN
            elif board.is_stalemate():
                game_result = DRAW
            elif board.is_insufficient_material():
                game_result = DRAW
            elif board.is_fifty_moves():
                game_result = DRAW
            elif board.is_repetition(3):
                game_result = DRAW
            
            board.print_()
            game_on = False
        
        elif board.is_fifty_moves() or board.is_repetition(3):
            game_result = DRAW
            board.print_()
            game_on = False
    
    return game_result



def main():
    """
    Основная функция для запуска игры с ботом
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ЗАГРУЗКА МОДЕЛИ
    model = ChessModel().to(device)
    model_path = "chess_model_best.pt"
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            iteration = checkpoint.get('iteration', '?')
            print(f"Model loaded from {model_path} (iteration {iteration})")
        else:
            model.load_state_dict(checkpoint)
            print(f"Model loaded from {model_path}")
    
    except FileNotFoundError:
        print(f"Model file {model_path} not found!")
        print("   Starting with untrained model")
    
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Starting with untrained model")
    
    model.eval()
    
    # НАСТРОЙКИ ИГРЫ
    print("\n" + "=" * 50)
    print("BOT GAME SETTINGS")
    print("=" * 50)
    
    bot_color = input("Bot plays white? (y/n): ").lower() == 'y'
    
    try:
        sims = int(input("MCTS simulations (default 100): ") or "100")
    except ValueError:
        sims = 100
    
    try:
        temp = float(input("Temperature (default 0.5): ") or "0.5")
    except ValueError:
        temp = 0.5
    
    # ЗАПУСК ИГРЫ
    board = Chess()
    
    while True:
        print("\n" + "=" * 50)
        print("🎮 NEW GAME")
        print("=" * 50)
        
        game_result = ubGame(
            board=board,
            model=model,
            bot_plays_white=bot_color,
            simulations=sims,
            temperature=temp
        )
        
        if game_result == EXIT:
            print("\nGoodbye!")
            break
        
        print("\n" + "=" * 50)
        if game_result == WHITE_WIN:
            print("WHITE WINS!  1 - 0")
        elif game_result == BLACK_WIN:
            print("BLACK WINS!  0 - 1")
        elif game_result == DRAW:
            print("DRAW!  1/2 - 1/2")
        print("=" * 50)
        
        if input("\nPlay again? (y/n): ").lower() != 'y':
            print("\nGoodbye!")
            break
        
        board.reset_()
        model.eval()



if __name__ == "__main__":
    main()
