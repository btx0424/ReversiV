import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from multiprocessing import Pool
from torch.optim import Adam
from networks import *
from tqdm import tqdm
from ReversiZero import AI, place, test_place, game_ended, find_move
from ReversiTime import AI as Baseline

import numpy as np

def train(model: nn.Module, examples, epoch=15, batch_size=32):
    print('training')
    device = torch.device('cuda')
    model.cuda()
    optimizer = Adam(model.parameters())
    batch_num = int(len(examples)/batch_size +1)

    for e in range(epoch):
        print('Epoch: %d'%e)
        np.random.shuffle(examples)
        t = tqdm(range(batch_num), desc='Training Net')
        for _ in t:
            sample_ids = np.random.randint(len(examples), size=batch_size)
            state, value = list(zip(*[examples[i] for i in sample_ids]))
            state = torch.tensor(state, device=device, dtype=torch.float)
            value = torch.tensor(value, device=device, dtype=torch.float)

            score = model(state).squeeze()
            loss = F.mse_loss(score, value)
            t.set_postfix(loss=float(loss))
            loss.backward()
            optimizer.step()
        
def fun(model):
    def eval(board):
        moves = find_move(board, 1)
        for move in moves:
            board[move]=0.5
        return model(board)
    return eval        

def save_checkpoint(model: nn.Module, folder='checkpoints', name='checkpoint.pth'):
    path = os.path.join(folder, name)
    torch.save(model.state_dict(), path)

def initboard():
    board = np.zeros((8, 8), dtype=np.int)
    board[3][4] = board[4][3] = 1
    board[3][3] = board[4][4] = -1
    return board

def play(AI1, AI2, g=2):
    print('baseline test')
    w = 0
    l = 0
    AI1.color, AI2.color = -1, 1
    player, next_player = AI1, AI2 
    for _ in range(g):
        board = initboard()
        for _ in tqdm(range(70)):
            player.go(board)
            if player.candidate_list:
                move = player.candidate_list[-1]
                assert place(board, move[0], move[1], player.color)
            player, next_player = next_player, player
            if game_ended(board): break
        if  np.sum(board)*AI1.color>0: w+=1
        else: l+=1
    
    player, next_player = AI2, AI1    
    for _ in range(g):
        board = initboard()
        for _ in tqdm(range(70)):
            player.go(board)
            if player.candidate_list:
                move = player.candidate_list[-1]
                assert place(board, move[0], move[1], player.color)
            player, next_player = next_player, player
            if game_ended(board): break
        if  np.sum(board)*AI1.color>0: w+=1
        else: l+=1
    return w, l

def excute_episode(net):
    print('self-play')

    player, next_player = AI(color=-1, fn=fun(net), time_out=3.0), AI(color=1, fn=fun(net), time_out=3.0)
    states = []
    board = initboard()
    for _ in tqdm(range(70)):
        canonical_board = board*player.color
        player.go(board)
        moves = player.candidate_list
        if moves:
            for move in moves: canonical_board[move]=0.5
            move = moves[-1]
            assert place(board, move[0], move[1], player.color)
        states.append((canonical_board, player.color))
        if game_ended(board): 
            winner_color = 1 if board.sum()>0 else -1  
            break
        else:
            player, next_player = next_player, player
         
    results =  [(board, 1 if color==winner_color else -1) for board, color in states]
    # augment using symmetry
    results.extend([(np.fliplr(board), value) for board, value in list(results)])  
    results.extend([(np.flipud(board), value) for board, value in list(results)])   
    results.extend([(board.T, value) for board, value in list(results)])  
    return results

def learn(iteration=5, episode=0):
    net = RevNet2()
    net.load_state_dict(torch.load('best.pth'))
    pnet = RevNet2()
    examples = []

    for i in range(iteration):
        print('Iter: %d'%i)
        for e in range(episode):
            print('Episode: %d'%e)
            with Pool() as pool:
                try:
                    multiple_results = [pool.apply_async(excute_episode, (net, )) for worker in range(4)]
                    for result in multiple_results:
                        examples.extend(result.get())
                except KeyboardInterrupt:
                    pool.terminate()
        
        
        torch.save(net.state_dict(), 'temp.pth')
        pnet.load_state_dict(torch.load('temp.pth'))
        train(net, examples) 

        w,l = play(AI(fn=fun(net)), AI(fn=fun(pnet)))
        print('played against the previous version, %d:%d'%(w,l))
        if w/(w+l)>0.6:
            torch.save(net.state_dict(), 'best.pth')

        w,l = play(AI(fn=fun(net)), Baseline(time_out=4.0))
        print('played against the baseline, %d:%d'%(w,l))
        if i%4==0:
            # examples = examples[-4800:]
            torch.save(examples, 'history_examples%d.pth'%i)
        
if __name__=='__main__':
    learn(10, 6)
    
    # net = NNet()
    # examples = excute_episode(net)
    # print(examples, len(examples))

    # net = NNet()
    # net.load_state_dict(torch.load('best.pth'))

    # w, l = play(AI(fn=fun(net), time_out=3.0), AI(time_out=3.5), g=2)
    # print(w, l)
        
