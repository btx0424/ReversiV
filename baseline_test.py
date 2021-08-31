from baselines.Reversi import AI as AlphaBeta
from baselines.ReversiTime import AI as MCTSTimed
from baselines.HBC import AI as HBC
from baselines.ReversiZeroSImple import AI as ZeroSimple

from ReversiZero import AI as Zero

import torch
import numpy as np

from networks import RevNet2
from tqdm import tqdm
from utils import *
import time

def printBoard(chessboard):
    print('  0 1 2 3 4 5 6 7')
    for i in range(8):
        print(str(i),end=' ')
        for j in range(8):
            if chessboard[i][j]==1:
                print('W',end=' ')
            if chessboard[i][j]==0:
                print('*',end=' ')
            if chessboard[i][j]==-1:
                print('B',end=' ')
        print()

def playgame(P1, P2):
    player1, name1 = P1
    player2, name2 = P2

    player1.color=-1
    player2.color=1

    board = np.zeros((8, 8), dtype=np.int)
    board[3][4] = board[4][3] = 1
    board[3][3] = board[4][4] = -1
    peak_time = 0
    player = player1

    stats = []
    for _ in tqdm(range(70)):
        time_start = time.perf_counter()
        player.go(board)
        stats.append(player.count)
        move = player.candidate_list
        time_took = time.perf_counter()-time_start
        if time_took>peak_time:
            peak_time = time_took
            by = player.color
            at = np.count_nonzero(board)-3
        if move:
            assert place(board, move[-1], player.color)
        player = player1 if player==player2 else player2
        # printBoard(board)
        if game_ended(board): break
    print(stats)   
    assert game_ended(board)
    print('%s vs %s: %d, peak time=%f by %s at %d'%(name1,name2,board.sum(),peak_time,by,at))
    print(board)
    return board.sum()

def fun(model):
    def eval(board):
        moves = find_move(board, 1)
        for move in moves:
            board[move]=0.5
        return model(board)
    return eval 


players = [
    (MCTSTimed(time_out=5.0),'MCTSTimed'),
    (HBC(), 'HBC'),
    (ZeroSimple(),'Simple'),
    (Zero(),'Zero'),
    (AlphaBeta(),'AI'),
]


playgame(players[0],players[1])
playgame(players[1],players[0])

