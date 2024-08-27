from torch import nn
import torch
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, embed_channel,kernel_size, stride, padding):
        super(Embedding, self).__init__()
        self.embedding = nn.Conv2d(1, 
                                   embed_channel, 
                                   kernel_size=kernel_size, 
                                   stride =stride, 
                                   padding= padding)
        
    def forward(self, x):
        x = self.embedding(x)
        return x

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        


class BasicCausalModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BasicCausalModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    