import chess
# all_ = 0
# board = [[0]*8 for _ in range(8)]
# for x in range(8):
#     for y in range(8):
#         diag = sum([min(x, y), min(7-x, y), min(x, 7-y), min(7-x, 7-y)])
#         x_, y_ = 7, 7         
#         kn = [0] * 8

#         if x == 0:
#             for i in range(4):
#                 kn[i] += 1
#         elif x == 1:
#             for i in range(4):
#                 kn[i] += 1
#             kn[4] += 1
#             kn[7] += 1
#         elif x == 6:
#             for i in range(4, 8):
#                 kn[i] += 1
#             kn[0] += 1
#             kn[3] += 1
#         elif x == 7:
#             for i in range(4, 8):
#                 kn[i] += 1
#         else:
#             for i in range(8):
#                 kn[i] += 1
                
#         if y == 0:
#             kn[0] += 1
#             kn[1] += 1
#             kn[6] += 1
#             kn[7] += 1
#         elif y == 1:
#             kn[0] += 1
#             kn[1] += 1
#             kn[6] += 1
#             kn[7] += 1
#             kn[2] += 1
#             kn[5] += 1
#         elif y == 6:
#             kn[1] += 1
#             kn[6] += 1
#             kn[2] += 1
#             kn[3] += 1
#             kn[4] += 1
#             kn[5] += 1
#         elif y == 7:
#             kn[2] += 1
#             kn[3] += 1
#             kn[4] += 1
#             kn[5] += 1
#         else:
#             for i in range(8):
#                 kn[i] += 1

#         for i in range(8):
#             kn[i] //= 2
            
#         pawns_reshape = 0
#         if y == 1 or y == 6:
#             if x == 0:
#                 pawns_reshape += 4*2
#             elif x == 7:
#                 pawns_reshape += 4*2
#             else:
#                 pawns_reshape += 4*2
            
#         result = x_ + y_ + diag  + pawns_reshape + sum(kn)
#         board[y][x] = result
#         all_ += result
#         print(f"({x}, {y}): {result}")

# for line in board[::-1]:
#     print(*line)
# print(all_)
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