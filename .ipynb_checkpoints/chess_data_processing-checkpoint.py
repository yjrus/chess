import torch
import chess
from chess_game import Chess, uu_game, user_input
from model import ChessModel

EXIT = -101
NEW_GAME = -102
CANCEL_MOVE = -103
HISTORY = -104

WHITE_WIN = -200
BLACK_WIN = -201
DRAW = -202

board = Chess()

class MCTS:
    def __init__(self, move: chess.Move, p: int = 0):
        super(MCTS, self).__init__()
        self.move = move
        self.P = p
        
        self.Q = 0
        self.N = 0
        
        self.children = []
        self.parent = None
        self.tensor = None
        self.mask = None



def boardToTensor(board: Chess) -> torch.Tensor:
    result = torch.zeros(20, 8, 8)
    """
    channels:
    0) whose move
    1-12) positions
    13-16) castlings
    17) en passant
    18) ply
    19) white material
    20) black material
    """
    info = board.fen().split()
    #fen: [0]positions [1]whose_move [2]castlings [3]en_passant [4]ply [5]move

    # whose move
    if info[1] == "b":
        result[0] = 1

    # positions + materials
    pieceToChannel = {
        "P": 1, "R": 2, "N": 3, "B":4, "Q": 5, "K": 6,
        "p": 7, "r": 8, "n": 9, "b":10, "q": 11, "k": 12
    }
    
    pieceToMaterial = {
        "p": 1, "n": 3, "b":3, "r": 5, "q": 9, "k": 0
    }
    
    r, c = 7, 0
    for row in info[0].split("/"):
        for piece in row:
            if piece.isdigit():
                c += int(piece)
                continue
            else:
                result[pieceToChannel[piece], r, c] = 1
                pl = piece.lower()
                if piece == pl:
                    result[20] += pieceToMaterial[pl] / 39.0
                else:
                    result[19] += pieceToMaterial[pl] / 39.0
            c += 1
        r -= 1
        c = 0
    
    # castlings
    castlingMap = {"K": 0, "Q": 1, "k": 2, "q": 3}
    for symb in info[2]:
        if symb != "-":
            result[13 + castlingMap[symb]] = 1

    # en passant
    if info[3] != "-": 
        c = ord(info[3][0]) - ord("a")
        r = int(info[3][1]) - 1
        result[3, r, c] = 1

    # ply
    result[15] = int(info[4]) / 100.0
    
    return result



def movesMask(board: Chess) -> torch.Tensor:
    result = torch.zeros(4184, 1)
    for connection in range(4096):
        fromCeil = connection // 64
        toCeil = connection % 64
        if fromCeil == toCeil:
            continue
        move = chess.Move(fromCeil, toCeil)        
        if board.is_legal(move):
            result[connection] = 1
            
    for startConnection in range(88):
        connection = startConnection
        fromRow, toRow = 1, 0
        newPiece = chess.QUEEN
        if connection >= 44:
            fromRow, toRow = 6, 7
            connection %= 44
        if connection % 2 == 0:
            newPiece = chess.KNIGHT
        connection += 2
        connection //= 2
        fromColumn = connection // 3
        if connection % 3 == 0:
            toColumn = fromColumn - 1
        elif connection % 3 == 1:
            toColumn = fromColumn
        else:
            toColumn = fromColumn + 1
        fromCeil = 8 * fromRow + fromColumn
        toCeil = 8 * toRow + toColumn
        move = chess.Move(fromCeil, toCeil, newPiece)
        if board.is_legal(move):
            result[4096 + startConnection] = 1
    return result



def PUCT(node: MCTS) -> MCTS:
    # пока что тут просто заглушка
    return node.children[0]


    
def MCTS_simulations(root: MCTS, model: ChessModel,
                     board: Chess, count: int = 100) -> str:
    for i in range(800):
        currentNode = root
        while currentNode.N != 0:
            currentNode = PUCT(currentNode)
            
            
        if currentNode.N == 0:
            currentNode.N += 1
            t_0 = boardToTensor(board)
            currentNode.tensor = t_0
            t_1 = torch.zeros(20, 8, 8)
            t_2 = torch.zeros(20, 8, 8)
            t_3 = torch.zeros(20, 8, 8)
            tmp = currentNode
            if tmp.parent is not None:
                tmp = tmp.parent
                t_1 = tmp.tensor
                if tmp.parent is not None:
                    tmp = tmp.parent
                    t_2 = tmp.tensor
                    if tmp.parent is not None:
                        tmp = tmp.parent
                        t_3 = tmp.tensor
            forModelTensor = torch.cat(t_0, t_1, t_2, t_3).unsqueeze(0)
            
            mask = movesMask(board)
            currentNode.mask = mask
            policy, value = model(forModelTensor)
            currentNode.Q = value
            target = policy + mask
            for startCeil in range(64):
                for dstCeil in range(64):
                    moveInd = startCeil * 64 + dstCeil
                    if target[moveInd] != 0:
                        newNode = MCTS(move = chess.Move(startCeil, dstCeil), p = target[moveInd], parent = currentNode)
                        current.cildren.append(newNode)
            # updating Node's data
            pass
            
            
    
    
    
    

    

    

    
        
                

