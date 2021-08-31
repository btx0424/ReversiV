import torch
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np
class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
    def forward(self, s):
        s = s.view(-1,64) 
        weights0 = F.leaky_relu(self.fc1(s))
        weights1 = F.leaky_relu(self.fc2(s))
        score = torch.sum(s *(weights0 + weights1), dim=1)
        s = torch.tanh(score)
        return s
class CNNet(nn.Module):
    def __init__(self):
        super(CNNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(8,8))
    def forward(self, s):
        s = s.view(-1,1,8,8)
        w = F.leaky_relu(self.conv1(s).view(-1, 64))
        s = s.view(-1, 64)
        s = torch.tanh(torch.sum(s*w, dim=1)) 
        return s

class RevNet2(nn.Module):
    def __init__(self, state_dict=None):
        super(RevNet2, self).__init__()
        
        self.weight1 = nn.Linear(in_features=64, out_features=64)
        self.weight2 = nn.Linear(in_features=64, out_features=64)
        self.fc0 = nn.Linear(in_features=64, out_features=2)
        self.fc1 = nn.Linear(in_features=64, out_features=2)
        self.fc2 = nn.Linear(in_features=64, out_features=2)

        self.score = nn.Linear(in_features=6, out_features=1)
        if state_dict:
            self.load_state_dict(state_dict)
    def forward(self, s):
        s = s.view(-1,64) 
        score0 = F.leaky_relu(self.fc0(s * F.leaky_relu(self.weight1(s))))
        score1 = F.leaky_relu(self.fc1(s * F.leaky_relu(self.weight2(s))))
        score2 = F.leaky_relu(self.fc2(s))

        s = torch.cat((score0, score1, score2), dim=1)
        
        return torch.tanh(self.score(s))
    def predict(self, board):
        with torch.no_grad():
            board = torch.from_numpy(board.astype(np.float32))
            v = self(board)
        return v.numpy()[0]