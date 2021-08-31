import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
import numpy as np
import os
from utils import *
import logging
from torch.utils.tensorboard import SummaryWriter
from networks import *
from pickle import Pickler, Unpickler
# writer = SummaryWriter()
log = logging.getLogger(__name__)

from Reversi import AI as Baseline

def train(nnet: nn.Module, examples, epochs = 100, batch_size = 32):
    device = torch.device('cuda')
    nnet.cuda()
    optimizer = Adam(nnet.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2, last_epoch=-1)

    test_cases = torch.load('test_cases')
    S, V = list(zip(*[test_cases[i] for i in range(len(test_cases))]))
    S = torch.tensor(S, device=device)
    T = torch.tensor(V, device=device)
    # nnet.train()
    for epoch in range(1,epochs+1  ):
        print('EPOCH ::: ' + str(epoch))
        
        np.random.shuffle(examples)
        batch_count = int(len(examples)/batch_size)
        t = tqdm(range(batch_count), desc='Training Net')
        for _ in t:
            sample_ids = np.random.randint(len(examples), size=batch_size)
            states, values = list(zip(*[examples[i] for i in sample_ids]))

            states = torch.tensor(states, device=device, dtype=torch.float)
            target = torch.tensor(values, device=device, dtype=torch.float)

            score = nnet(states).squeeze()
            # print(target.shape, score.shape)
            # exit()
            # loss = F.mse_loss(score, target)
            loss = torch.sum((target-score)**2)/target.shape[0]
            t.set_postfix(loss=float(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
         
        scheduler.step()
        
        with torch.no_grad():
            sample_ids = np.random.randint(len(examples), size=len(examples))
            states, values = list(zip(*[examples[i] for i in sample_ids]))
            states = torch.tensor(states, device=device, dtype=torch.float )
            target = torch.tensor(values, device=device, dtype=torch.float)
            pred = nnet(states).squeeze()
            acc=(pred.sign() == target.sign()).sum()/float(len(sample_ids))
            print('training acc:',acc)
        
        # if epoch%10==0:
        #     with torch.no_grad():
        #         _, pred = torch.max(nnet(S).squeeze(), 1)
        #         acc=(pred == T).sum()/float(len(test_cases))
        #         print('test acc:',acc) 

def save_checkpoint(model: nn.Module, folder='checkpoints', name='checkpoint.pth'):
    path = os.path.join(folder, name)
    torch.save(model.state_dict(), path)

def excute_episode(AI1, AI2, num_eps=1):
    winner_states = []
    loser_states = []
    for _ in tqdm(range(num_eps), desc='playing'):
        AI1.color, AI2.color = -1,1
        ws, ls= play_game(AI1, AI2)
        winner_states.extend(ws)
        loser_states.extend(ls)
        AI1.color, AI2.color = 1,-1
        ws, ls= play_game(AI2, AI1)
        winner_states.extend(ws)
        loser_states.extend(ls)

    for states in (winner_states, loser_states):
        tmp = states.copy()
        tmp.extend([np.flip(s) for s in states])
        tmp.extend([np.flip(s,0) for s in states])
        tmp.extend([np.flip(s,1) for s in states])
        T = [s.T for s in tmp]
        states.clear()
        states.extend(tmp)
        states.extend(T)
    
    result = [(state.astype(np.float32), 1) for state in winner_states]
    result.extend([(state.astype(np.float32), 0) for state in loser_states]) 

    return result

def learn(nnet, examples):
    
    train(nnet, examples)
    save_checkpoint(nnet, name = 'checkpoint.pth')
    
LOAD_MODEL = False            
MODEL_PATH = 'checkpoints/'
LOAD_EXAMPLES = True

if __name__=='__main__':
    nnet = RevNet2()
    if LOAD_MODEL:
        nnet.load_state_dict(torch.load('checkpoints/checkpoint.pth'))
    device = torch.device('cuda')
    nnet.cuda()

    examples = []

    if LOAD_EXAMPLES:
        examples.extend(torch.load('Example_History.pth'))
    
    # examples = preprocess(examples)
    # print(examples[0])
    np.random.shuffle(examples)

    try:
        learn(nnet, examples)
    except KeyboardInterrupt:
        save_checkpoint(nnet, name = 'interruptRev.pth')

    
    
    

    
    


