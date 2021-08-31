import numpy as np
from tqdm import tqdm 

DIR = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)) # 方向向量
INF = 1e8
def find_move(board, color):
    moves = []
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:
                newBoard = board.copy()
                if test_place(newBoard, i, j, color):
                    moves.append((i, j))
    return moves
def test_place(board, x, y, color):
    for d in range(8):
        i = x + DIR[d][0]
        j = y + DIR[d][1]
        while 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == -color:
            i += DIR[d][0]
            j += DIR[d][1]
        if 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == color:
            i -= DIR[d][0]
            j -= DIR[d][1]
            if not(i == x and j == y):
                return True
    return False

def game_ended(board):
    if not find_move(board, 1) and not find_move(board, -1): 
        return True
    else:
        return False

def place(board, move, color):
    x, y = move
    if x < 0:
        return False
    board[x][y] = color
    valid = False
    for d in range(8):
        i = x + DIR[d][0]
        j = y + DIR[d][1]
        while 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == -color:
            i += DIR[d][0]
            j += DIR[d][1]
        if 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == color:
            while True:
                i -= DIR[d][0]
                j -= DIR[d][1]
                if i == x and j == y:
                    break
                valid = True
                board[i][j] = color
    return valid

def play_game(player1, player2):
    board = np.zeros((8, 8), dtype=np.int)
    board[3][4] = board[4][3] = 1
    board[3][3] = board[4][4] = -1
    player = player1

    while np.count_nonzero(board)-3<51:
        player.go(board)
        move = player.candidate_list
        if move:
            assert place(board, move[-1], player.color)
        player = player1 if player==player2 else player2
        if game_ended(board): break
    winner_states = []
    loser_states = []

    if judge(board, 1):
        winner_states.append(board.copy())
    else:
        loser_states.append(board.copy())
    
    if judge(board, -1):
        winner_states.append(board.copy()*-1)
    else:
        loser_states.append(board.copy()*-1)
    return winner_states, loser_states
 
def judge(board, myColor):
    
    def max_value(board, color, alpha, beta):
        moves = find_move(board, color)
        if not moves:
            if not find_move(board, -color):
                return np.sum(board) * myColor
            else: 
                return min_value(board.copy(), -color, alpha, beta)
        v = -INF
        for x, y in moves:
            next_board = board.copy()
            assert place(next_board, (x, y), color)
            v = max(v, min_value(next_board, -color, alpha, beta))
            if v>= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(board, color, alpha, beta):
        moves = find_move(board, color)
        if not moves:
            if not find_move(board, -color):
                return np.sum(board) * myColor
            else: 
                return max_value(board.copy(), -color, alpha, beta)
        v = INF
        for x, y in moves:
            next_board = board.copy()
            assert place(next_board, (x, y), color)
            v = min(v, max_value(next_board, -color, alpha, beta))
            if v<=alpha:
                return v
            beta = min(beta, v)
        return v

    best_score = -INF
    beta = INF

    for x, y in find_move(board, myColor):
        next_board = board.copy()
        place(next_board, (x, y), myColor)
        v = min_value(next_board, -myColor, best_score, beta)
        if v>0: 
            return True
        if v> best_score:
            best_score = v
    return False

if __name__=='__main__':

    AI1 = RandomAI(color=-1)
    AI2 = RandomAI(color=1)
    winner_states, loser_states= play_game(AI1, AI2)
    print(winner_states, loser_states)
    for states in winner_states:
        assert judge(states, 1)

    for states in loser_states:
        assert not judge(states, 1)