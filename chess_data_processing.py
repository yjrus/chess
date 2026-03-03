import torch
from torch.distributions import Dirichlet
import chess
from math import sqrt
from chess_game import Chess
from model import ChessModel



def boardToTensor(board: Chess) -> torch.Tensor:
    """
    Преобразует доску в тензор 21x8x8
    """
    result = torch.zeros(21, 8, 8)
    
    info = board.fen().split()
    
    # Канал 0: очередь хода
    if info[1] == "b":
        result[0] = 1
    
    # Каналы 1-12, 19-20: позиции фигур, материалы
    piece_to_channel = {
        "P": 1, "R": 2, "N": 3, "B": 4, "Q": 5, "K": 6,
        "p": 7, "r": 8, "n": 9, "b": 10, "q": 11, "k": 12
    }

    piece_to_material = {
        "p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0
    }

    r, c = 7, 0
    for row in info[0].split("/"):
        for piece in row:
            if piece.isdigit():
                c += int(piece)
                continue
            else:
                result[piece_to_channel[piece], r, c] = 1
                pl = piece.lower()
                if piece == pl:
                    result[20] += piece_to_material[pl] / 39.0
                else:
                    result[19] += piece_to_material[pl] / 39.0
            c += 1
        r -= 1
        c = 0
    
    # Каналы 13-16: рокировки
    castling_map = {"K": 0, "Q": 1, "k": 2, "q": 3}
    for symb in info[2]:
        if symb != "-":
            result[13 + castling_map[symb]] = 1
    
    # Канал 17: взятие на проходе
    if info[3] != "-":
        c = ord(info[3][0]) - ord("a")
        r = int(info[3][1]) - 1
        result[17, r, c] = 1
    
    # Канал 18: счетчик полуходов
    result[18] = int(info[4]) / 100.0
    
    return result



def codeToPromotion(code: int) -> chess.Move:
    """
    Преобразует код (0-87) в promotion ход
    """
    code %= 88
    from_row, to_row = 1, 0
    promotion = chess.QUEEN
    
    if code >= 44:
        from_row, to_row = 6, 7
    if code % 2 == 0:
        promotion = chess.KNIGHT
    
    code %= 44
    code += 2
    code //= 2
    from_column = code // 3
    
    if code % 3 == 0:
        to_column = from_column - 1
    elif code % 3 == 1:
        to_column = from_column
    else:
        to_column = from_column + 1
    
    from_square = from_row * 8 + from_column
    to_square = to_row * 8 + to_column
    
    return chess.Move(from_square, to_square, promotion)



def promotionIndex(move: chess.Move) -> int:
    """
    Возвращает индекс для promotion хода (0-87) в соответствии с codeToPromotion
    """
    if move.promotion is None:
        return -1
    
    from_row = move.from_square // 8
    to_row = move.to_square // 8
    col_from = move.from_square % 8
    col_to = move.to_square % 8
    
    is_white_promotion = (from_row == 1 and to_row == 0)
    is_black_promotion = (from_row == 6 and to_row == 7)
    
    if not (is_white_promotion or is_black_promotion):
        return -1
    
    color_offset = 0 if is_white_promotion else 44
    direction = col_to - col_from
    
    if col_from == 0 and direction == -1:
        return -1
    if col_from == 7 and direction == 1:
        return -1
    
    # Вычисление базового кода позиции
    if col_from == 0:
        pos_code = 0 if direction == 0 else 1
    elif col_from == 7:
        pos_code = 0 if direction == -1 else 1
    else:
        dir_code = direction + 1
        pos_code = (col_from - 1) * 3 + dir_code + 2
    
    # Определение типа превращения (0 - конь, 1 - ферзь)
    promo_type = 0 if move.promotion == chess.KNIGHT else 1
    
    result = color_offset + pos_code * 2 + promo_type
    
    if 0 <= result < 88:
        return result
    else:
        print(f"Warning: invalid promotion index {result} for move {move}")
        return 0



def movesMask(board: Chess) -> torch.Tensor:
    """
    Создает маску легальных ходов (4184, 1)
    """
    result = torch.zeros(4184, 1)
    
    for connection in range(4096):
        from_square = connection // 64
        to_square = connection % 64
        
        if from_square == to_square:
            continue
            
        move = chess.Move(from_square, to_square)
        
        try:
            if board.is_legal(move):
                result[connection] = 1
        except:
            continue
    
    for code in range(88):
        move = codeToPromotion(code)
        if board.is_legal(move):
            result[4096 + code] = 1
    
    return result



class MCTS:
    """
    Узел дерева MCTS
    """
    def __init__(self,
                 move: chess.Move,
                 p: int = 0,
                 parent = None):
        
        super(MCTS, self).__init__()
        self.move = move
        self.P = p
        
        self.Q = 0.0
        self.W = 0.0
        self.N = 0
        
        self.children = []
        self.is_extended = False
        self.parent = parent
        self.tensor = None
        self.mask = None



def puct(node: MCTS) -> MCTS:
    """
    Selection по формуле PUCT
    """
    best_score = -float('inf')
    best_child = None
    c_puct = 1.5
    
    for child in node.children:
        U = c_puct * child.P * (sqrt(node.N + 1e-6) / (1 + child.N))
        score = U + child.Q
        
        if score > best_score:
            best_score = score
            best_child = child
    
    return best_child



def dirichletNoise(mask: torch.Tensor,
                   policy: torch.Tensor) -> torch.Tensor:
    """
    Добавляет Dirichlet noise к policy для root узла
    """
    if mask.device != policy.device:
        mask = mask.to(policy.device)
    
    alpha = 0.3
    eps = 0.25
    
    if policy.dim() > 1:
        policy = policy.flatten()
    
    legal_moves = int(mask.sum().item())
    concentration = torch.full((legal_moves,), alpha)
    noise = Dirichlet(concentration).sample()
    
    noise_idx = 0
    for policy_idx in range(4096 + 88):
        if policy[policy_idx].item() != 0:
            policy[policy_idx] = policy[policy_idx] * (1 - eps) + noise[noise_idx] * eps
            noise_idx += 1
    
    return policy.flatten()



def selectMoveWithTemperature(root: MCTS, temperature: float):
    """
    Выбор хода с температурой
    """
    if not root.children:
        return None
    
    visits = torch.tensor(
        [child.N for child in root.children],
        dtype=torch.float32
    )
    
    if temperature <= 1e-6:
        idx = torch.argmax(visits).item()
        return root.children[idx].move
    
    visits = visits ** (1.0 / temperature)
    probs = visits / visits.sum()
    idx = torch.multinomial(probs, 1).item()
    
    return root.children[idx].move



def MCTS_simulations(root: MCTS,
                     model: ChessModel,
                     board: Chess,
                     count: int = 100,
                     batch_size: int = 64):
    """
    Запускает симуляции MCTS для поиска лучшего хода
    """
    device = next(model.parameters()).device
    model.eval()
    
    VIRTUAL_LOSS = 1.0
    simulations_done = 0
    root_stack_len = len(board.move_stack)
    
    while simulations_done < count:
        
        current_batch = min(batch_size, count - simulations_done)
        leaf_nodes = []
        leaf_paths = []
        
        # SELECTION
        for _ in range(current_batch):
            node = root
            path = [node]
            
            while node.is_extended and node.children:
                node = puct(node)
                board.push(node.move)
                path.append(node)
                
                # Virtual loss
                node.N += 1
                node.W -= VIRTUAL_LOSS
            
            leaf_nodes.append(node)
            leaf_paths.append(path)
            
            # Откат к root
            while len(board.move_stack) > root_stack_len:
                board.pop()
        
        # PREPARE STATES
        states = []
        terminal_values = []
        
        for i, node in enumerate(leaf_nodes):
            
            # Восстановление позиции листа
            path = leaf_paths[i]
            for n in path[1:]:
                board.push(n.move)
            
            if board.is_game_over():
                
                result = board.result()
                if result == "1-0":
                    winner = chess.WHITE
                elif result == "0-1":
                    winner = chess.BLACK
                else:
                    winner = None
                
                if winner is None:
                    terminal_values.append(0.0)
                else:
                    terminal_values.append(
                        1.0 if winner == board.turn else -1.0
                    )
                
                states.append(None)
            
            else:
                
                tensors = []
                moves_to_restore = []
                
                # Сбор предыдущих состояний
                for step in range(min(4, len(board.move_stack))):
                    tensors.append(boardToTensor(board))
                    
                    if len(board.move_stack) > root_stack_len:
                        move = board.pop()
                        moves_to_restore.append(move)
                
                # Дополнение нулями до 4 состояний
                while len(tensors) < 4:
                    if tensors:
                        tensors.append(torch.zeros_like(tensors[0]))
                    else:
                        tensors.append(torch.zeros(21, 8, 8))
                
                # Восстановление позиции
                for move in reversed(moves_to_restore):
                    board.push(move)
                
                state = torch.cat(tensors, dim=0)
                states.append(state)
                terminal_values.append(None)
            
            # Откат к root
            while len(board.move_stack) > root_stack_len:
                board.pop()
        
        # MODEL INFERENCE
        non_terminal_idx = [
            i for i, s in enumerate(states) if s is not None
        ]
        
        if non_terminal_idx:
            batch_states = torch.stack(
                [states[i] for i in non_terminal_idx]
            ).to(device)
            
            with torch.no_grad():
                logits, values = model(batch_states)
            
            values = values.squeeze(1)
        else:
            logits = None
            values = None
        
        value_ptr = 0
        
        # EXPANSION + BACKPROP
        for i, node in enumerate(leaf_nodes):
            
            path = leaf_paths[i]
            
            # Восстановление позиции листа
            for n in path[1:]:
                board.push(n.move)
            
            if terminal_values[i] is not None:
                value = terminal_values[i]
            
            else:
                value = values[value_ptr].item()
                policy_logits = logits[value_ptr]
                value_ptr += 1
                
                mask = movesMask(board).flatten().to(device)
                mask = mask.to(policy_logits.dtype)
                policy = torch.softmax(policy_logits, dim=0)
                policy = policy * mask
                
                if node is root:
                    policy = dirichletNoise(mask, policy)
                
                if policy.sum() > 0:
                    policy /= policy.sum()
                else:
                    policy = mask / mask.sum()
                
                # Расширение узла
                if not node.is_extended:
                    node.is_extended = True
                    
                    for move in board.legal_moves:
                        
                        if move.promotion is None:
                            idx = move.from_square * 64 + move.to_square
                        else:
                            idx = 4096 + promotionIndex(move)
                        
                        p = policy[idx].item()
                        
                        if p > 1e-12:
                            child = MCTS(
                                move=move,
                                p=p,
                                parent=node
                            )
                            node.children.append(child)
            
            # Backpropagation
            v = value
            
            for n in reversed(path):
                n.W += VIRTUAL_LOSS
                n.N += 1
                n.W += v
                n.Q = n.W / n.N
                v = -v
            
            # Откат к root
            while len(board.move_stack) > root_stack_len:
                board.pop()
        
        simulations_done += current_batch
    
    if not root.children:
        return None
    
    best_child = max(root.children, key=lambda n: n.N)
    return best_child.move
