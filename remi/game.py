import random
from typing import Callable
from typing import Tuple
import numpy as np

class Board:
    def __init__(self, board:'Board'=None):
        if board is None:
            # columns are first, then rows
            # 0 is empty, 1 is player 1, 2 is player 2
            self.pieces = np.zeros((7, 6), dtype=int)
            self.valid = True
        else:
            self.pieces = board.pieces.copy()
            self.valid = board.valid

def MakeMove(board: Board, col: int, player: int) -> 'Board':
    newBoard = Board(board)
    if (newBoard.pieces[col] == 0).sum() == 0:
        newBoard.valid = False
        return newBoard
    lowestIndex = np.where(newBoard.pieces[col] == 0)[0][0]
    newBoard.pieces[col][lowestIndex] = player
    return newBoard

def FindWinner(board: Board) -> int:
    # check columns
    for col in range(7):
        for i in range(3):
            if (board.pieces[col][i:i+4] == 1).all():
                return 1
            if (board.pieces[col][i:i+4] == -1).all():
                return -1
    # check rows
    for row in range(6):
        for i in range(4):
            if (board.pieces[i:i+4, row] == 1).all():
                return 1
            if (board.pieces[i:i+4, row] == -1).all():
                return -1
    # check diagonals
    for col in range(4):
        for row in range(3):
            area = board.pieces[col:col+4, row:row+4]
            flippedArea = np.fliplr(area)
            if (area.diagonal() == 1).all():
                return 1
            if (area.diagonal() == -1).all():
                return -1
            if (flippedArea.diagonal() == 1).all():
                return 1
            if (flippedArea.diagonal() == -1).all():
                return -1
    return 0

def PrintBoard(board: Board):
    for row in range(5, -1, -1):
        for col in range(7):
            if board.pieces[col][row] == 0:
                print('. ', end='')
            elif board.pieces[col][row] == 1:
                print('X ', end='')
            elif board.pieces[col][row] == -1:
                print('O ', end='')
        print()

# return (move, evaluation)
def EvalBoard(board: Board, player: int, evalFunc: Callable[[Board], int], depth:int=1) -> Tuple[int, int]:
    # base case
    if depth == 1:
        possibleMoves = [MakeMove(board, i, player) for i in range(7)]
        evaluations = [evalFunc(move) for move in possibleMoves]
        winningMoves = [i for i in range(7) if evaluations[i] == player]
        if len(winningMoves) != 0:
            return (random.choice(winningMoves), player)
        neutralMoves = [i for i in range(7) if evaluations[i] == 0]
        if len(neutralMoves) != 0:
            return (random.choice(neutralMoves), 0)
        losingMoves = [i for i in range(7) if evaluations[i] == -player]
        return (random.choice(losingMoves), -player)
    
    # recursive case
    possibleMoves = [MakeMove(board, i, player) for i in range(7)]
    evaluations = [EvalBoard(move, -player, evalFunc, depth-1) for move in possibleMoves]
    winningMoves = [i for i in range(7) if evaluations[i][1] == player]
    if len(winningMoves) != 0:
        return (random.choice(winningMoves), player)
    neutralMoves = [i for i in range(7) if evaluations[i][1] == 0]
    if len(neutralMoves) != 0:
        return (random.choice(neutralMoves), 0)
    losingMoves = [i for i in range(7) if evaluations[i][1] == -player]
    return (random.choice(losingMoves), -player)

def PlayPlayerVersusPlayer():
    board = Board()
    player = 1
    while True:
        print()
        PrintBoard(board)
        print("Player", player, "turn")
        try:
            col = int(input("Enter column: "))
            if col < 0 or col > 6:
                print("\nInvalid column: please enter a number between 0 and 6")
                continue
        except ValueError:
            print("\nInvalid input: please enter a number between 0 and 6")
            continue
        newBoard = MakeMove(board, col, player)
        if not newBoard.valid:
            print("\nInvalid move: column is full")
            continue
        board = newBoard
        winner = FindWinner(board)
        if winner != 0:
            print()
            PrintBoard(board)
            print("Player", winner, "wins!")
            break
        player = -player

def PlayPlayerVersusAI(aiEvalFunc: Callable[[Board], int], aiDepth:int=1):
    board = Board()
    player = 1
    while True:
        if player == 1:
            print()
            PrintBoard(board)
            print("Player", player, "turn")
            try:
                col = int(input("Enter column: "))
                if col < 0 or col > 6:
                    print("\nInvalid column: please enter a number between 0 and 6")
                    continue
            except ValueError:
                print("\nInvalid input: please enter a number between 0 and 6")
                continue
            newBoard = MakeMove(board, col, player)
            if not newBoard.valid:
                print("\nInvalid move: column is full")
                continue
            board = newBoard
        else:
            move = EvalBoard(board, player, aiEvalFunc, aiDepth)[0]
            newBoard = MakeMove(board, move, player)
            if not newBoard.valid:
                print("AI made an invalid move")
                break
            board = newBoard
        winner = FindWinner(board)
        if winner != 0:
            print()
            PrintBoard(board)
            print("Player", winner, "wins!")
            break
        player = -player

def EvalBoardRandom(board: Board) -> int:
    return random.randint(-1, 1)

PlayPlayerVersusAI(EvalBoardRandom, aiDepth=1)
