import math

X = 'X'
O = 'O'

user_move = O
AI_move = X

def get_turn(board):
    numX = 0
    numO = 0
    for row in board:
        for pos in row:
            if pos == X:
                numX += 1
            elif pos == O:
                numO += 1
    return X if numX == numO else O

def getMoves(board):
    moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] not in [X, O]:
                moves.append((i,j))
    return moves

def getWinner(board):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != '.':
            return board[i][0]
    for j in range(3):
        if board[0][j] == board[1][j] == board[2][j] != '.':
            return board[0][j]
    # diagonals & cross diagonals
    if board[0][0] == board[1][1] == board[2][2] != '.':
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != '.':
        return board[0][2]
    return None

def isOver(board):
    if getWinner(board) is not None:
        return True
    for row in board:
        for cell in row:
            if cell == '.':
                return False
    return True

def printBoard(board):
    for row in board:
        print(" ".join(row))
    print()

def minimax(board, turn, alpha, beta):
    if isOver(board):
        winner = getWinner(board)
        if winner == X:
            return 1
        elif winner == O:
            return -1
        return 0
    if turn == X:
        maxVal = -math.inf
        for move in getMoves(board):
            new_board = [row[:] for row in board]
            new_board[move[0]][move[1]] = turn
            val = minimax(new_board, O, alpha, beta)
            maxVal = max(val, maxVal)
            alpha = max(alpha, maxVal)
            if beta <= alpha:
                break
        return maxVal
    else:
        minVal = math.inf
        for move in getMoves(board):
            new_board = [row[:] for row in board]
            new_board[move[0]][move[1]] = turn
            val = minimax(new_board, X, alpha, beta)
            minVal = min(val, minVal)
            beta = min(beta, minVal)
            if beta <= alpha:
                break
        return minVal

def bestMove(board):
    best_score = -math.inf
    move = None
    for i, j in getMoves(board):
        new_board = [row[:] for row in board]
        new_board[i][j] = X
        score = minimax(new_board, O, -math.inf, math.inf)
        if score > best_score:
            best_score = score
            move = (i, j)
    return move

def runGame():
    board = [
        ['.', '.', '.'],
        ['.', '.', '.'],
        ['.', '.', '.']
    ]
    while not isOver(board):
        printBoard(board)
        turn = get_turn(board)
        if turn == O:
            i, j = map(int, input("Enter row col: ").split())
            if board[i][j] == '.':
                board[i][j] = O
            else:
                print("Invalid move")
                continue
        else:
            i,j = bestMove(board)
            board[i][j] = X
            print(f"AI played: ({i},{j})")
    printBoard(board)
    winner = getWinner(board)
    if winner:
        print("Winner:", winner)
    else:
        print("Draw")

runGame()
