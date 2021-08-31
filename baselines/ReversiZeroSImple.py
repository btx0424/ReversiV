
import numpy as np
import random
from time import perf_counter
import threading
WEIGHT20 = np.array([
       [ 1.986, -0.045,  0.055,  0.037,  0.037,  0.055, -0.045,  1.971],
       [-0.045, -0.304, -0.113, -0.056, -0.059, -0.116, -0.306, -0.045],
       [ 0.055, -0.118,  0.004, -0.011, -0.013,  0.006, -0.116,  0.055],
       [ 0.037, -0.067, -0.013,  0.018,  0.021, -0.013, -0.066,  0.037],
       [ 0.037, -0.06 , -0.011,  0.022,  0.016, -0.02 , -0.066,  0.037],
       [ 0.055, -0.117,  0.013, -0.02 , -0.012,  0.004, -0.111,  0.055],
       [-0.045, -0.297, -0.112, -0.072, -0.066, -0.118, -0.303, -0.045],
       [ 1.99 , -0.045,  0.055,  0.037,  0.037,  0.055, -0.045,  1.979]],
      dtype=np.float16)
WEIGHT50 = np.array([
    [ 0.98541204,  0.04035845,  0.04384775,  0.04191742,  0.04185729, 0.04486757,  0.04413301,  0.97116454],
    [ 0.04848264, -0.30446803, -0.11258063, -0.05608972, -0.05857577,-0.11605463, -0.30616653,  0.0405198 ],
    [ 0.04045199, -0.11835686,  0.00369619, -0.01073905, -0.013084  , 0.005706  , -0.11639149,  0.03769541],
    [ 0.05197595, -0.06659195, -0.01282388,  0.01832268,  0.0205201 ,-0.01321658, -0.06568969,  0.04408597],
    [ 0.05241211, -0.0599413 , -0.01134043,  0.02166829,  0.01608053,-0.01993317, -0.06629244,  0.04791857],
    [ 0.05366062, -0.11717497,  0.01329587, -0.01950666, -0.01219502,0.0036302 , -0.11091427,  0.04376664],
    [ 0.05751975, -0.29662156, -0.11206735, -0.0714942 , -0.06593978,-0.11826274, -0.3029509 ,  0.04539823],
    [ 0.99039419,  0.03853371,  0.0364874 ,  0.03694624,  0.04121141,0.03857323,  0.04190149,  0.97900956],
])

COEFS = [
    np.array([[ 2.0938e+00,  1.8140e-01, -8.3847e-03,  2.0923e-01],
       [-2.1436e-01,  2.0667e-01,  1.0315e-01,  1.4722e-01],
       [ 7.1777e-02,  5.5206e-02,  7.5745e-02, -4.7852e-02],
       [-1.2932e-02,  7.8796e-02,  1.0834e-01,  4.7638e-02],
       [ 6.2317e-02, -7.8247e-02,  3.0960e-02,  9.7778e-02],
       [ 1.4542e-02,  4.9988e-02, -3.3386e-02,  1.0492e-01],
       [ 2.2058e-01, -7.2021e-01,  1.1707e-01,  6.7871e-02],
       [ 1.0498e-01,  5.1641e+00,  2.0288e-01, -3.4424e-02],
       [-2.0532e-01,  1.4246e-01,  1.0834e-01,  2.4072e-01],
       [-6.3184e-01, -8.9294e-02, -1.9409e-02, -9.8511e-02],
       [-1.2903e-01, -1.2842e-01, -1.4656e-02,  4.8676e-03],
       [-9.0393e-02, -5.3345e-02,  4.3373e-03, -9.7733e-03],
       [-6.4514e-02, -1.0468e-01, -7.7896e-03,  1.1299e-02],
       [-1.3379e-01, -1.4124e-01,  7.3318e-03, -1.7822e-02],
       [-2.0679e-01, -6.8701e-01, -1.4148e-01, -4.2633e-02],
       [ 1.5466e-01, -6.7920e-01,  2.2961e-01,  1.0547e-01],
       [ 7.2815e-02, -3.7384e-02,  7.7942e-02,  7.6294e-02],
       [-1.2769e-01,  4.6425e-03, -1.0780e-02, -1.4136e-01],
       [-5.6610e-02,  2.8397e-02,  7.4768e-02,  3.4607e-02],
       [-4.4098e-02, -5.0934e-02, -1.0101e-02,  2.4887e-02],
       [-2.5223e-02, -4.9957e-02,  2.4551e-02, -8.5602e-03],
       [ 1.3687e-02, -7.2998e-02,  6.5422e-03,  5.6458e-02],
       [-5.9509e-02, -1.4880e-01, -1.4587e-01, -1.2344e-02],
       [-6.3848e-04,  5.8014e-02,  6.3354e-02,  9.7595e-02],
       [-1.3092e-02,  4.1199e-02,  1.0931e-01,  9.5276e-02],
       [-8.8257e-02, -1.8250e-02, -8.6060e-03, -7.0129e-02],
       [-3.2196e-02,  2.8854e-02, -3.8929e-03, -6.3599e-02],
       [ 4.1992e-02,  6.6910e-03,  6.7627e-02,  1.5187e-04],
       [ 1.4954e-02,  9.0256e-03,  2.1378e-02,  6.0730e-02],
       [ 7.8812e-03, -2.6505e-02, -9.4177e-02, -7.4615e-03],
       [-3.8239e-02, -1.2030e-01, -8.8257e-02, -7.9117e-03],
       [ 5.0262e-02, -7.3303e-02,  7.9712e-02,  1.1414e-01],
       [ 6.0852e-02,  8.6975e-02,  2.5375e-02, -9.6985e-02],
       [-6.2622e-02,  7.7629e-03, -1.2199e-02, -1.1786e-01],
       [-1.5656e-02, -5.8556e-03,  2.1591e-02, -4.1260e-02],
       [ 1.9089e-02,  5.9082e-02,  2.1729e-02,  1.6495e-02],
       [ 5.2948e-02,  1.7288e-02,  1.9440e-02,  1.8051e-02],
       [-2.8717e-02, -5.2307e-02, -2.4155e-02,  3.4149e-02],
       [-3.0014e-02, -6.2805e-02, -1.3489e-01, -2.3453e-02],
       [ 8.0750e-02,  7.9285e-02, -1.0406e-01,  2.5024e-02],
       [ 1.8814e-02,  8.2458e-02, -2.6749e-02,  5.8716e-02],
       [-1.1823e-01, -1.6602e-02,  1.3718e-02, -1.4478e-01],
       [ 2.0340e-02,  4.9530e-02,  1.3428e-02, -8.6182e-02],
       [ 2.9030e-03, -9.4681e-03, -1.0168e-01, -3.4058e-02],
       [-1.6403e-02,  3.8177e-02, -2.5040e-02, -7.3669e-02],
       [ 5.1117e-02,  1.1887e-02, -8.8989e-02,  4.3068e-03],
       [-8.4045e-02, -1.1432e-01, -1.5063e-01,  2.2675e-02],
       [ 1.6861e-02,  8.0994e-02,  8.8074e-02, -4.0619e-02],
       [ 2.3218e-01,  6.6833e-02,  1.1847e-01, -8.2568e-01],
       [-1.9653e-01, -4.5074e-02, -1.3818e-01, -7.8760e-01],
       [-4.8859e-02, -1.9073e-02, -1.4319e-01, -1.5991e-01],
       [-2.6215e-02, -1.5898e-03, -8.6182e-02, -1.3318e-01],
       [-2.6901e-02, -2.4841e-02, -1.3501e-01, -8.3679e-02],
       [-8.2275e-02,  1.4549e-02, -1.5747e-01, -1.4087e-01],
       [-1.3745e-01, -1.4685e-01, -8.1689e-01, -1.7322e-01],
       [ 1.4600e-01,  2.0740e-01, -8.2471e-01,  1.3379e-01],
       [ 1.2463e-01, -4.1840e-02,  2.2351e-01,  7.0938e+00],
       [ 1.5283e-01,  9.3689e-02,  2.2864e-01, -8.0859e-01],
       [ 5.6696e-04,  7.5256e-02,  7.2083e-02,  5.6580e-02],
       [ 4.4495e-02,  9.6924e-02,  8.2397e-02, -1.0327e-01],
       [ 7.9285e-02,  1.7288e-02, -1.0638e-01,  8.4473e-02],
       [ 3.1891e-03, -3.9673e-02,  8.4229e-02,  7.6599e-02],
       [ 1.4355e-01,  1.1713e-01, -8.1934e-01,  2.1277e-01],
       [ 1.0445e-02,  2.0007e-01,  7.2734e+00,  2.2156e-01]],
      dtype=np.float16),
    np.array([[0.36882489],
       [0.28592947],
       [0.29918055],
       [0.28342508]])  
]
INTERCEPTS = [np.array([0.15191616, 0.22264317, 0.24840539, 0.2470838 ]), np.array([0.01413649])]

DIR = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)) # 方向向量

INF = 1e10

def place(board, x, y, color):
        if x < 0:
            return True
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
def test_place(board, x, y, color):
    for d in range(8):
        i = x + DIR[d][0]
        j = y + DIR[d][1]
        while 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == -color:
            i += DIR[d][0]
            j += DIR[d][1]
        if  0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == color:
            i -= DIR[d][0]
            j -= DIR[d][1]
            if not(i == x and j == y):
                return True
    return False

def find_move(board, color):
    moves = []
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:
                if test_place(board, i, j, color):
                    moves.append((i, j))
    return moves

def game_ended(board):
    if find_move(board, 1) or find_move(board,-1):
        return 0
    else:
        if board.sum()==0:
            return -1
        return board.sum() 

def eval(board):
    turn = np.count_nonzero(board)-3
    if turn < 40:
        return np.sum(board * WEIGHT20)
    return np.matmul(np.tanh(np.matmul(board.reshape(1,64), COEFS[0]) + INTERCEPTS[0]), COEFS[1])+INTERCEPTS[1]

def house_keeping(Qsa, Nsa, Ns, Es, Vs, turn):
    try:
        for s in list(Vs.keys()):
            if np.count_nonzero(np.fromstring(s,dtype=np.int))<turn:
                del Vs[s]
                del Ns[s]
                del Es[s]
        for s,a in list(Nsa.keys()):
            if np.count_nonzero(np.fromstring(s,dtype=np.int))<turn:
                del Nsa[(s,a)]
                del Qsa[(s,a)] 
    except KeyError:
        pass
    
#don't change the class name
class AI(object):
    #chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size=8, color=-1, time_out=4.5, fn=eval):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out-0.35
        self.candidate_list = []

        self.repeated = 0
        self.state_searched = 0
        self.cpuct = 1.2

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

# The input is current chessboard.
    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.repeated = 0
        self.state_searched = 0
        self.candidate_list.clear()
        self.candidate_list = find_move(chessboard, self.color)
        if self.candidate_list:
            start = perf_counter()
            turn = np.count_nonzero(chessboard)-3
            if turn>46:
                try:
                    s = (chessboard*self.color).tostring()
                    self.candidate_list.sort(key=lambda move: self.Nsa[(s, move)])
                except KeyError:
                    self.candidate_list.sort(key=lambda move: WEIGHT50[move[0],move[1]])
                self.candidate_list.append(self.alpha_beta(chessboard))
            else:
                self.candidate_list.append(self.MCTS(chessboard*self.color))
                threading.Thread(target=house_keeping, args=(self.Qsa, self.Nsa, self.Ns, self.Es, self.Vs, turn)).start()
            print('player=%d time=%f turn=%d state searched=%d repeated=%d'%(self.color,perf_counter()-start,turn,self.state_searched,self.repeated))
        
        
    def alpha_beta(self, board):
        def max_value(board, color):
            s= (board*color).tostring() 
            self.state_searched+=1
            if s in self.Vs:
                self.repeated+=1
            else:
                self.Vs[s] = find_move(board, color)
            moves = self.Vs[s]
            if not moves:
                if not find_move(board, -color):
                    return np.sum(board)*self.color # ?
                else: 
                    return min_value(board.copy(), -color)
            try:
                moves.sort(key=lambda move: self.Nsa[(s, move)], reverse=True)
            except KeyError:
                moves.sort(key=lambda move: WEIGHT50[move[0],move[1]], reverse=True)
            v = -INF
            for x, y in moves:
                next_board = board.copy()
                assert place(next_board, x, y, color)
                v = max(v, min_value(next_board, -color))
                if v>0:
                    return v
            return v
        def min_value(board, color):
            s= (board*color).tostring() 
            self.state_searched+=1
            if s in self.Vs:
                self.repeated+=1
            else:
                self.Vs[s] = find_move(board, color)
            moves = self.Vs[s]
            if not moves:
                if not find_move(board, -color):
                    return np.sum(board)*self.color
                else:
                    return max_value(board.copy(), -color)
            try:
                moves.sort(key=lambda move: self.Nsa[(s, move)], reverse=True)
            except KeyError:
                moves.sort(key=lambda move: WEIGHT50[move[0],move[1]], reverse=True)
            v = INF
            for x, y in moves:
                next_board = board.copy()
                assert place(next_board, x, y, color)
                v = min(v, max_value(next_board, -color))
                if v<0:
                    return v
            return v

        moves = find_move(board, self.color)
        s = (board*self.color).tostring()
        try:
            moves.sort(key=lambda move: self.Nsa[(s, move)], reverse=True)
        except KeyError:
            moves.sort(key=lambda move: WEIGHT50[move[0],move[1]], reverse=True)
        for x, y in moves:
            next_board = board.copy()
            assert place(next_board, x, y, self.color)
            v = min_value(next_board, -self.color)
            if v> 0:
                print('definite win!')
                return(x ,y)
        return moves[0]
    

    def MCTS(self, canonicalBoard):
        start_time = perf_counter()
        # search until run out of time
        count = 0
        while perf_counter() - start_time < self.time_out:
            self._search(canonicalBoard)
            count+=1
        s = canonicalBoard.tostring()
        
        moves = self.Vs[s]
        # values = [self.Qsa[(s, a)] for a in moves]
        # best_move = np.argmax(values)
        counts = [self.Nsa[(s, a)] for a in moves]
        best_move = np.argmax(counts)
        return moves[best_move]

    def _search(self, canonicalBoard):
        s = canonicalBoard.tostring()
        if s not in self.Es:
            self.Es[s] = game_ended(canonicalBoard)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ns:
            # leaf node
            v = eval(canonicalBoard)
            moves = find_move(canonicalBoard, 1)
            self.Vs[s] = moves
            self.Ns[s] = 0
            return -v
        
        moves = self.Vs[s]
        cur_best = -float('inf')
        best_act = (-1, -1)
        if moves: p = 1.0/len(moves)
        # pick the action with the highest upper confidence bound
        for a in moves:
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.cpuct * p * np.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
            else:
                u = self.cpuct * p * np.sqrt(self.Ns[s] + 1e-8)  
            if u > cur_best:
                cur_best = u
                best_act = a

        next_board = canonicalBoard.copy()
        assert place(next_board, best_act[0], best_act[1], 1)
        v = self._search(-next_board)
        a = best_act
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
    
        self.Ns[s] += 1
        return -v
            

    