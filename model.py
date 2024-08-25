import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, ):
        super(Embedding, self).__init__()
        self.embedding = nn.Conv2d(kernel_size=kernel_size, stride = stride, padding = padding)
        
    def forward(self, x):
        return self.embedding(x)

class Encoder(nn.Module):
    def __init__():
        
    def forward(self,x):
        return x
    
class Decoder(nn.Module):
    def __init__(self, decoder_dim,):
    
    def forward(self,x):
        return x
    
class Head(nn.Module):
    def __init__(decoder_dim, num_classes):
        
    def forward(self,x):
        
        return x
        

class ARC_Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = Embedding()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.head = Head